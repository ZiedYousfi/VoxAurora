use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use crate::whisper_integration::{
  burt_correct_text, clean_whisper_text, merge_separated_words_dawg_regex,
};

fn bench_clean_whisper_text(c: &mut Criterion) {
  let mut group = c.benchmark_group("text_cleanup");

  let sample_text = "[_BEG_] Aujourd'hui est un [_TT_42] jour  magnifique.";
  group.bench_function("clean_whisper_text", |b| {
    b.iter(|| clean_whisper_text(black_box(sample_text)))
  });

  group.finish();
}

fn bench_burt_correct_text(c: &mut Criterion) {
  let mut group = c.benchmark_group("text_correction");

  // Simple text with common errors
  let sample_text = "La voiture roule tres vite sur l'autoroute.";
  group.bench_function("burt_correct_text", |b| {
    b.iter(|| burt_correct_text(black_box(sample_text)))
  });

  group.finish();
}

fn bench_merge_separated_words(c: &mut Criterion) {
  let mut group = c.benchmark_group("word_merging");

  let test_cases = vec![
    ("aujourd hui", 2),
    ("bon jour à tous", 2),
    ("c est un exemple de phrase avec des mots séparés", 2),
    ("je m appelle jean et j habite à paris", 2)
  ];

  for (i, (text, max_merge)) in test_cases.iter().enumerate() {
    group.bench_with_input(
      BenchmarkId::new("merge_separated_words", i),
      &(text, max_merge),
      |b, &(text, max_merge)| {
        b.iter(|| merge_separated_words_dawg_regex(black_box(text), black_box(*max_merge)))
      }
    );
  }

  group.finish();
}

criterion_group!(
  benches,
  bench_clean_whisper_text,
  bench_burt_correct_text,
  bench_merge_separated_words
);
criterion_main!(benches);
