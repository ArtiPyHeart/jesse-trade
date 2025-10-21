// NRBO 性能基准测试

use criterion::{black_box, criterion_group, criterion_main, Criterion};

// TODO: 实现完整的NRBO后启用此测试
fn bench_nrbo_placeholder(c: &mut Criterion) {
    c.bench_function("nrbo_placeholder", |b| {
        b.iter(|| {
            black_box(1 + 1)
        });
    });
}

criterion_group!(benches, bench_nrbo_placeholder);
criterion_main!(benches);
