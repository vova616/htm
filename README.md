## A HTM Machine Learning library

A Rust implementation of [HTM](https://github.com/numenta/htmresearch-core)

Currently compatible with Java/(probably Python/C++ too) by using the same algo and random generator.

Performance are similar to C++.

Examples in examples folder, MNIST example got a score of 95% accuracy.

### WIP

- [x] SpatialPooler
- [x] SDRClassifier
- [ ] Encoders
    - [x] ScalarEncoder
    - [x] AdaptiveScalarEncoder
    - [x] DeltaEncoder
    - [ ] DateEncoder
- [x] TemporalMemory
