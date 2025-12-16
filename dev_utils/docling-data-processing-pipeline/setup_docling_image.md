

### Setup custom image with docling

Generate yaml pipeline definition
```commandline
python process_bucket_pipeline.py
```

Build image from `docling.Dockerfile` 
```commandline
podman build --arch amd64 -t quay.io/openshift_trial/custom_images:docling .
```
Login to quay.io
```commandline
podman login quay.io
```
Push image to quay.io registry
```commandline
podman push quay.io/openshift_trial/custom_images:docling
```
