/*!
 * Ripser Rust 性能基准测试
 *
 * 使用 Criterion 框架进行性能测试。
 */

use criterion::{criterion_group, criterion_main, Criterion};
use rust_indicators::ripser::core::BinomialCoeffTable;
use std::hint::black_box;

/// 基准测试：构建二项式系数表
fn benchmark_binomial_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("binomial_construction");

    // 小规模
    group.bench_function("n=20_k=10", |b| {
        b.iter(|| BinomialCoeffTable::new(black_box(20), black_box(10)))
    });

    // 中规模
    group.bench_function("n=50_k=25", |b| {
        b.iter(|| BinomialCoeffTable::new(black_box(50), black_box(25)))
    });

    // 最大安全规模
    group.bench_function("n=57_k=28", |b| {
        b.iter(|| BinomialCoeffTable::new(black_box(57), black_box(28)))
    });

    group.finish();
}

/// 基准测试：查询二项式系数
fn benchmark_binomial_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("binomial_get");

    let table = BinomialCoeffTable::new(100, 50);

    // 单次查询
    group.bench_function("single_query", |b| {
        b.iter(|| table.get(black_box(50), black_box(25)))
    });

    // 批量查询（模拟实际使用场景）
    group.bench_function("batch_query_100", |b| {
        b.iter(|| {
            for i in 0..100 {
                black_box(table.get(i, i / 2));
            }
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark_binomial_construction, benchmark_binomial_get);
criterion_main!(benches);
