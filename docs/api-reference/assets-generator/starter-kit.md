# Starter Kit

Asset generation has moved to
[`kfp-components`](https://github.com/kubeflow/pipelines-components/tree/main/components/training/autorag/shared/assets_generator).
Import from `kfp_components.components.training.autorag.shared.assets_generator`. The old `ai4rag.assets_generator` package has been removed without a compatibility shim.

See the [current documentation](https://github.com/kubeflow/pipelines-components/blob/main/components/training/autorag/shared/assets_generator/docs/starter-kit.md)
and [coordinated migration and runtime requirements](https://github.com/kubeflow/pipelines-components/blob/main/components/training/autorag/shared/assets_generator/README.md).
Generated notebooks and applications still depend on ai4rag.
