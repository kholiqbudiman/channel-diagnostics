# Dataset sources and experimental partitions

- **WaRP-C / author-provided data location:** https://www.kaggle.com/kholiqbudiman/datasets . This is an author-provided collection page, not a verified dataset-specific permalink. Exact dataset slug/version and its match to the retained manifests must be confirmed before public release. Do not assume the collection contains a new dataset authored for this study.
- **TrashNet:** original project https://github.com/garythung/trashnet . The authors currently link downloads at https://huggingface.co/datasets/garythung/trashnet . Use the retained experimental manifests to identify the actual files/partitions rather than adopting a different split by default.
- **Describable Textures Dataset (DTD):** https://www.robots.ox.ac.uk/~vgg/data/dtd/ . The study uses 3,760 images from train1+val1, repartitioned with split seeds 6301–6303 into fit/selection/diagnostic/evaluation roles. This is not the official test benchmark.

Respect each original dataset's terms and attribution. Raw images are not redistributed in this repository. Data membership and image hashes, where retained, are in the protocol JSON files under research_artifacts; WaRP-C labels use semantic parent folders rather than the incorrect legacy index mapping.
