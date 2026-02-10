```python
def fit(self):
    """
    The training loop of PPO.
    The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
    The light-weight advantage computation is done on the driver process.
    """
    logger = Tracking(
        project_name=self.config.trainer.project_name,
        experiment_name=self.config.trainer.experiment_name,
        default_backend=self.config.trainer.logger,
        config=self.config.to_dict(),
    )
    self.global_steps = 0

    # load checkpoint before doing anything
    self._load_checkpoint()

    # perform validation before training
    # currently, we only support validation using the reward_function.
    if self.val_reward_fn is not None and self.config.trainer.val_before_train:
        val_metrics = self._validate()
        pprint(f"Initial validation metrics: {val_metrics}")
        logger.log(data=val_metrics, step=self.global_steps)
        if self.config.trainer.val_only:
            return

    for _ in range(self.config.trainer.total_episodes):
        for batch_dict in self.train_dataloader:
            self.global_steps += 1
            if self.global_steps >= self.training_steps:
                break

            metrics = {}
            timing_raw = {}

            batch: DataProto = DataProto.from_single_dict(batch_dict)

            # Add temporal corruption for TAPO if enabled
            if self.config.algorithm.use_kl_prcp and "images" in batch.non_tensor_batch.keys():
                video_sequences = batch.non_tensor_batch["images"]
                corrupted_video_sequences = self._temporal_corruption(video_sequences)
                # Store corrupted sequences for later use
                batch.non_tensor_batch["aug_multi_modal_data"] = corrupted_video_sequences

            # pop those keys for generation
            if "pixel_values" in batch.non_tensor_batch.keys():
                gen_batch = batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["pixel_values", "image_grid_thw", "raw_prompt_ids", "images"],
                )
            else:
                gen_batch = batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )

            with _timer("step", timing_raw):
                # generate a batch
                with _timer("gen", timing_raw):  # wg: worker group
                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                if self.config.algorithm.adv_estimator == "remax":
                    with _timer("gen_max", timing_raw):
                        gen_baseline_batch = deepcopy(gen_batch)
                        gen_baseline_batch.meta_info["do_sample"] = False
                        gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                        batch = batch.union(gen_baseline_output)
                        reward_baseline_tensor = self.reward_fn(batch)
                        reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                        batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                        batch.batch["reward_baselines"] = reward_baseline_tensor

                        del gen_baseline_batch, gen_baseline_output

                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )
                # repeat to align with repeated responses in rollout
                batch = batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
                batch = batch.union(gen_batch_output)

                # balance the number of valid tokens on each dp rank.
                # Note that this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                # self._balance_batch(batch, metrics=metrics) # TODO: re-enable balance batch

                # compute global_valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # recompute old_log_probs
                with _timer("old_log_prob", timing_raw):
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    batch = batch.union(old_log_prob)

                # Compute augmented log probabilities for TAPO if enabled
                if self.config.algorithm.use_kl_prcp and "aug_multi_modal_data" in batch.non_tensor_batch.keys():
                    with _timer("aug_log_prob", timing_raw):
                        # Create a batch with corrupted video sequences for augmented log probability computation
                        aug_batch = deepcopy(batch)
                        # Replace original images with corrupted ones
                        aug_batch.non_tensor_batch["images"] = batch.non_tensor_batch["aug_multi_modal_data"]
                        # Also update pixel_values if they exist
                        if "pixel_values" in aug_batch.non_tensor_batch:
                            # You might need to convert corrupted videos to pixel values
                            # This depends on your implementation of _temporal_corruption
                            aug_batch.non_tensor_batch["pixel_values"] = self._convert_to_pixel_values(
                                batch.non_tensor_batch["aug_multi_modal_data"]
                            )
                        
                        aug_log_prob = self.actor_rollout_wg.compute_log_prob(aug_batch)
                        batch = batch.union(aug_log_prob)

                if self.use_reference_policy:
                    # compute reference log_prob
                    with _timer("ref", timing_raw):
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)

                # compute values
                if self.use_critic:
                    with _timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with _timer("adv", timing_raw):
                    # compute scores. Support both model and function-based.
                    # We first compute the scores using reward model. Then, we call reward_fn to combine
                    # the results from reward model and rule-based results.
                    if self.use_reward_model:
                        raise NotImplementedError

                    # we combine with rule-based rm
                    reward_tensor = self.reward_fn(batch)
                    batch.batch["token_level_scores"] = reward_tensor

                    # compute rewards. apply_kl_penalty if available
                    if not self.config.worker.actor.use_kl_loss:  # not grpo
                        batch, kl_metrics = apply_kl_penalty(
                            batch, 
                            kl_ctrl=self.kl_ctrl, 
                            kl_penalty=self.config.algorithm.kl_penalty,
                            # Add TAPO-specific KL penalty if enabled
                            use_tapo_kl=self.config.algorithm.use_kl_prcp
                        )
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # compute advantages, executed on the driver process
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=self.config.worker.rollout.n,
                    )

                # update critic
                if self.use_critic:
                    with _timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)

                # implement critic warmup
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    with _timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_wg.update_actor(batch)

                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with _timer("testing", timing_raw):
                        val_metrics: dict = self._validate()
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                    with _timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # collect metrics
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

    # perform validation after training
    if self.val_reward_fn is not None:
        val_metrics = self._validate()
        pprint(f"Final validation metrics: {val_metrics}")
        logger.log(data=val_metrics, step=self.global_steps)

    self._save_checkpoint()

```


```python
def _temporal_corruption(self, video_sequences):
    """
    Apply temporal corruption to video sequences for TAPO.
    
    Args:
        video_sequences: List of video sequences, each is [frame0, frame1, frame2, ...]
    
    Returns:
        List of corrupted video sequences
    """
    corrupted_sequences = []
    
    for video in video_sequences:
        # Choose a temporal corruption strategy
        # Here are some options:
        
        # Option 1: Random frame shuffling
        if self.config.algorithm.tapo_strategy == "shuffle":
            corrupted = list(video)
            random.shuffle(corrupted)
            corrupted_sequences.append(corrupted)
        
        # Option 2: Frame dropping (randomly drop some frames)
        elif self.config.algorithm.tapo_strategy == "drop":
            keep_prob = self.config.algorithm.tapo_keep_prob
            corrupted = [frame for frame in video if random.random() < keep_prob]
            # If we dropped all frames, keep at least one
            if len(corrupted) == 0:
                corrupted = [video[0]]
            corrupted_sequences.append(corrupted)
        
        # Option 3: Temporal reversal
        elif self.config.algorithm.tapo_strategy == "reverse":
            corrupted = list(reversed(video))
            corrupted_sequences.append(corrupted)
        
        # Option 4: Fixed interval sampling
        elif self.config.algorithm.tapo_strategy == "interval":
            interval = self.config.algorithm.tapo_interval
            corrupted = video[::interval]
            if len(corrupted) == 0:
                corrupted = [video[0]]
            corrupted_sequences.append(corrupted)
        
        # Default: No corruption (for testing)
        else:
            corrupted_sequences.append(video)
    
    return corrupted_sequences

def _convert_to_pixel_values(self, corrupted_video_sequences):
    """
    Convert corrupted video sequences to pixel values if needed.
    This will depend on your specific preprocessing pipeline.
    """
    # Implement based on your existing preprocessing
    # This is a placeholder - you'll need to adapt it to your codebase
    pixel_values = []
    for video in corrupted_video_sequences:
        # Process each frame in the video
        frame_pixels = []
        for frame in video:
            # Convert frame to tensor/processed format
            # This depends on your preprocessing
            processed_frame = self._preprocess_frame(frame)
            frame_pixels.append(processed_frame)
        pixel_values.append(torch.stack(frame_pixels))
    
    return torch.stack(pixel_values)

```

```python
def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", use_tapo_kl=False):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch["attention_mask"]
    response_mask = attention_mask[:, -response_length:]

    # Initialize total penalty as zero
    total_penalty = torch.zeros_like(response_mask, dtype=torch.float32)
    
    # Original KL penalty between ref_policy and current policy
    if "ref_log_prob" in data.batch.keys():
        kld = core_algos.kl_penalty(
            data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
        )  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
        
        # Add original KL penalty to total penalty
        total_penalty = total_penalty + beta * kld
        current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
        current_kl = torch.mean(current_kl, dim=0).item()
        
        # Update original KL controller
        kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
        
        # Store for metrics
        original_kl = current_kl
        original_beta = beta
    else:
        original_kl = 0.0
        original_beta = 0.0

    # TAPO-specific KL penalty if enabled
    tapo_kl = 0.0
    tapo_beta = 0.0
    if use_tapo_kl and "aug_log_probs" in data.batch.keys():
        # Compute KL divergence between original and augmented predictions
        # KL(original || augmented) = orig_log_probs - aug_log_probs
        tapo_kld = data.batch["old_log_probs"] - data.batch["aug_log_probs"]
        tapo_kld = tapo_kld * response_mask
        
        # Get TAPO KL weight
        if hasattr(kl_ctrl, 'tapo_kl_weight'):
            tapo_beta = kl_ctrl.tapo_kl_weight
        else:
            # Default TAPO weight if not specified
            tapo_beta = 0.1
            
        # Add TAPO penalty to total penalty
        total_penalty = total_penalty + tapo_beta * tapo_kld
        
        # Compute TAPO KL metrics
        tapo_kl = masked_mean(tapo_kld, mask=response_mask, axis=-1)
        tapo_kl = torch.mean(tapo_kl, dim=0).item()

    # Apply total penalty to rewards
    token_level_rewards = token_level_scores - total_penalty

    data.batch["token_level_rewards"] = token_level_rewards

    # Prepare metrics
    metrics = {
        "critic/kl": original_kl, 
        "critic/kl_coeff": original_beta,
    }
    
    # Add TAPO metrics if applicable
    if use_tapo_kl and "aug_log_probs" in data.batch.keys():
        metrics.update({
            "tapo/kl": tapo_kl,
            "tapo/kl_coeff": tapo_beta,
            "tapo/total_penalty": masked_mean(total_penalty, mask=response_mask, axis=-1).mean().item(),
        })

    return data, metrics
```


```yaml
algorithm:
  use_kl_prcp: true  # Enable TAPO
  tapo_kl_weight: 0.1  # Weight for TAPO penalty
  kl_penalty: "kl"  # Type of KL penalty (unchanged)
  
  # Optional: Add TAPO-specific parameters
  tapo_corruption_type: "freeze"  # Options: freeze, shuffle, drop, reverse
  tapo_keep_prob: 0.7  # For drop strategy
  tapo_interval: 2  # For interval strategy
```


```python
# In your fit() method, inside the training loop:

# 1. Check if we should use TAPO and add corrupted data
if self.config.algorithm.use_kl_prcp and "images" in batch.non_tensor_batch.keys():
    video_sequences = batch.non_tensor_batch["images"]
    corrupted_video_sequences = self._temporal_corruption(video_sequences)
    # Store corrupted sequences for later use
    batch.non_tensor_batch["aug_multi_modal_data"] = corrupted_video_sequences

# 2. Compute original log probabilities
with _timer("old_log_prob", timing_raw):
    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
    batch = batch.union(old_log_prob)

# 3. Compute augmented log probabilities for TAPO
if self.config.algorithm.use_kl_prcp and "aug_multi_modal_data" in batch.non_tensor_batch.keys():
    with _timer("aug_log_prob", timing_raw):
        # Create a batch with corrupted video sequences
        aug_batch = deepcopy(batch)
        # Replace original images with corrupted ones
        aug_batch.non_tensor_batch["images"] = batch.non_tensor_batch["aug_multi_modal_data"]
        # Also update pixel_values if they exist
        if "pixel_values" in aug_batch.non_tensor_batch:
            aug_batch.non_tensor_batch["pixel_values"] = self._convert_to_pixel_values(
                batch.non_tensor_batch["aug_multi_modal_data"]
            )
        
        aug_log_prob = self.actor_rollout_wg.compute_log_prob(aug_batch)
        batch = batch.union(aug_log_prob)

# 4. Compute rewards and apply KL penalty (including TAPO penalty)
with _timer("adv", timing_raw):
    reward_tensor = self.reward_fn(batch)
    batch.batch["token_level_scores"] = reward_tensor

    # compute rewards with TAPO penalty
    if not self.config.worker.actor.use_kl_loss:  # not grpo
        batch, kl_metrics = apply_kl_penalty(
            batch, 
            kl_ctrl=self.kl_ctrl, 
            kl_penalty=self.config.algorithm.kl_penalty,
            use_tapo_kl=self.config.algorithm.use_kl_prcp,
            tapo_kl_weight=self.config.algorithm.tapo_kl_weight
        )
        metrics.update(kl_metrics)
    else:
        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
```


