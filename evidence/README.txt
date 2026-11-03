NUMERICAL EVIDENCE

scores/ contains ten compressed NumPy archives. Load them with
numpy.load(path, allow_pickle=False). All scores use the same direction:
smaller values are accepted first. No source images, image filenames, image
paths, prompts, person identities, or dataset copies are included.

efficientnet_DeepShield / efficientnet_OpenFake
  10,000 / 4,000 stored decisions of the principal EfficientNetV2-M detector.
  All 26 archived selection methods are included, with raw and calibrated
  margins for its six views. DeepShield is in-domain held-out. OpenFake is the
  cross-dataset test relative to the explicitly documented fitting sources.

references_DeepShield / references_OpenFake
  The same principal evaluation images, reused retrospectively. Ten selection
  methods, the deterministic two-class logits, and the 30 native-head dropout
  logit pairs are stored. The six-draw result uses the first six of those same
  30 draws. Float32 storage of logits is lossless relative to the archived
  measurements; score reconstruction casts them back to float64.

clip_DeepShield / clip_OpenFake
resnet50_DeepShield / resnet50_OpenFake
  Initial and adapted detector outputs on the same principal test images,
  after adaptation using the principal detector's training manifest. Each
  detector has its own separately fitted full/JPEG/resize/confidence selector.
  These cohorts are retrospective reuse, not new independent test images.

independent_DeepShield / independent_OpenFake
  1,500 / 6,000 new-image ablation decisions comparing the previously frozen
  full selector with resize-only. This experiment tests four locked replacement
  conditions. Only OpenFake AURC passes; the original decision is retain_full.

ARRAY FIELDS

errors        1 if the original deterministic detector decision is wrong.
labels        Where supplied, 0 real and 1 synthetic.
prediction    Original deterministic detector class, where supplied.
score__NAME   Stored error-ranking score of method NAME.
tie_key       Original label-independent SHA-256 tie digest. Its lexicographic
              order is preserved exactly for fixed accepted-count metrics.
image_key     Namespace-hashed sample identifier. It is not an image checksum.
group_index   Opaque integer provenance-group index. For the first eight
              bundles, indices preserve the original sorted-group ordering.
              Independent-ablation indices derive from the existing opaque
              CSV labels and do not claim the original bootstrap index order.

Group definitions reflect the recorded experimental provenance. A group index
is not evidence that all possible dependencies or historical training overlap
have been ruled out. The independent OpenFake ablation has 43 connected groups,
including one group of 3,412 images; group-sensitivity results are included.

expected/ stores archived point outcomes and contrasts for each score bundle.
bootstrap/ stores numerical bootstrap draws where these were retained.
reports/ stores original aggregate results, including non-favorable results.
protocols/ stores release copies of the relevant locked settings. The principal
protocol is a field-selected summary that omits embedded source-file inventories.
Other local paths are redacted. Original protocol/checkpoint hashes identify
the unmodified private archive; they do not hash the release summary itself.

datasets.json fixes the score and expected-result checksums. source_files.json
records original source checksums and the export transformations. The release
manifest checks the files actually distributed in this repository.

Original source records, detector training images, pretrained/adapted detector
checkpoints, and third-party dataset annotations are not redistributed here.
Public-provider references and acquisition guidance are documented elsewhere
in the repository. The package verifies the reported analysis from stored
measurements; it does not independently regenerate those measurements without
the separately obtained images and detector artifacts.
