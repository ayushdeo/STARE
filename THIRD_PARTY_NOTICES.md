# Third-party notices

## SOFisher

`artifacts/sofish_seed222_5m/` holds network weights and TensorBoard logs from a native training run of SOFisher
(https://github.com/Heathcliff-Ng/SOFisher), and `sofisher_resume/apply_resume.py` patches a local SOFisher clone to
add checkpoint and resume support. SOFisher is distributed under the MIT License:

```
MIT License

Copyright (c) 2024 Heathcliff Ng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Data and models not redistributed

HEST-1k (`MahmoodLab/hest`) and UNI2-h (`MahmoodLab/UNI2-h`) are gated on Hugging Face under their own terms. This
repository contains no HEST data files or UNI weights. `results/` holds derived statistics and, in
`sample_candidates.json`, a few metadata fields per selected slide (identifier, organ, platform, pixel size, spot
count), the same information listed in the paper's slide table.
