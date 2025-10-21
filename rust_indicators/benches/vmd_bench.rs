// VMD 性能基准测试

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array1;

// TODO: 实现完整的VMD后启用此测试
fn bench_vmd_placeholder(c: &mut Criterion) {
    c.bench_function("vmd_placeholder", |b| {
        b.iter(|| {
            black_box(1 + 1)
        });
    });
}

criterion_group!(benches, bench_vmd_placeholder);
criterion_main!(benches);
