# Split Inference Demo with ResNet18 Classification of the ImageNet database

A pre-trained ResNet18 model is split into two separate models — client (Alice) and server (Bob).

<img src="public/split_inference_diagram.pnh">

Your uploaded image is sent through the client model which returns the split activations of a cut layer within the full model.

These activations are sent to a server where the remaining of the inference is completed. The output tensor is then sent back to the client.

This allows you to perform inference on a model present on a third party server while preserving your image's privacy.

## Available Scripts

In the project directory, you can run:

### `npm start`

Runs app locally (be sure to change the server model URL in `App.tsx` if needed)

### `npm deploy`

Deploys app to production (https://splitlearning.mit.edu/SplitLearningInference)