# **Data for PAPO**

## **Training Data**
We adapt [TIGER-Lab/ViRL39K](https://huggingface.co/datasets/TIGER-Lab/ViRL39K) and [FanqingM/MMK12](https://huggingface.co/datasets/FanqingM/MMK12) to train **PAPO**:
- `PAPO/papo_virl39k_train`: [Hugging Face Dataset](https://huggingface.co/datasets/PAPO/papo_virl39k_train)
- `PAPO/papo_mm_eureka_test`: [Hugging Face Dataset](https://huggingface.co/datasets/PAPO/papo_mm_eureka_test)

## **Evaluation Data**
We adapted 8 different multimodal reasoning datasets to evaluate **PAPO**, which are further splitted into `General Reasoning` and `Vision-Dependent Reasoning` evaluation datasets:
- **General Reasoning**
    - `hiyouga/geometry3k`: [Hugging Face Dataset](https://huggingface.co/datasets/hiyouga/geometry3k), [Data Source](https://github.com/lupantech/InterGPS)
    - `AI4Math/MathVista`: [Hugging Face Dataset](https://huggingface.co/datasets/AI4Math/MathVista)
    - `We-Math/We-Math`: [Hugging Face Dataset](https://huggingface.co/datasets/We-Math/We-Math)
    - `FanqingM/MMK12`: [Hugging Face Dataset](https://huggingface.co/datasets/FanqingM/MMK12)
    - `AI4Math/MathVerse`: [Hugging Face Dataset](https://huggingface.co/datasets/AI4Math/MathVerse)
- **Vision-Dependent Reasoning**
    - `lscpku/LogicVista`: [Hugging Face Dataset](https://huggingface.co/datasets/lscpku/LogicVista)
    - `BUAADreamer/clevr_count_70k`: [Hugging Face Dataset](https://huggingface.co/datasets/BUAADreamer/clevr_count_70k)
    - `MMMU/MMMU_Pro`: [Hugging Face Dataset](https://huggingface.co/datasets/MMMU/MMMU_Pro)
    - `MathVerse_V` (vision-dependent subset): Adapted from [AI4Math/MathVerse](https://huggingface.co/datasets/AI4Math/MathVerse)