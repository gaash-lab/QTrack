# Copyright 2026 - RMOT
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def freeze_future_frames(frames):
    """
        Structure of frames: List of Tensors with shape (B, 3, H, W)
        Freezes all frames except the first one by copying the first frame.
        frames will be a list , for example: [frame_0, frame_1, frame_2, ..., frame_n]
        After freezing, frames will be: [frame_0, frame_0, frame_0, ..., frame_0]
    """
    frozen = [frames[0]]
    for _ in frames[1:]:
        frozen.append(frames[0])    
    return frozen