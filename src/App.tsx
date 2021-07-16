
import { Component } from 'react';
import {Tensor, InferenceSession} from 'onnxjs';
import ndarray from "ndarray";
import ops from 'ndarray-ops';
import './App.css';

interface State {
  loading: string, // Loading/status message displayed to user
  modelLoaded: boolean, // flag to decide whether client model needs to be loaded -- needs to be loaded only during the first inference
  class: string,  // output of resnet model, identified ImageNet class of the image
  imageURL: string
}

interface Props {}

// the client model will be downloaded and used by ONNX
const RESNET18_CLIENT_MODEL_URL = 'http://splitlearning.mit.edu/SplitLearningInference/resnet18_client.onnx';
// the server model will be sent activations through a POST request
const RESNET18_SERVER_MODEL_URL = 'http://matlaber10.media.mit.edu:5000/inference';

class App extends Component<Props, State> {

  session = new InferenceSession({ backendHint: 'cpu' });

  constructor(props: Props) {
    super(props);
    this.state = {
        loading: null,
        modelLoaded: false,
        class: null,
        imageURL: null
    };
  }

  /**
   * Preprocessing (resize, crop, rearrange, normalize, totensor) image from canvas for ResNet inference
   * @param img: HTMLImageElement
   * @returns 
   */
  preprocessImage = (img: HTMLImageElement) => {
    // scale image so that the smaller dimension is 224 px
    var orgWidth = img.width;
    var orgHeight = img.height;
    var resizedHeight = 224;
    var resizedWidth = 224;
    if (orgWidth > orgHeight) {
      resizedWidth = (orgWidth / orgHeight) * 224;
    } else if (orgWidth < orgHeight) {
      resizedHeight = (orgHeight / orgWidth) * 224;
    }
    console.log("Resized (" + orgWidth + "px, " + orgHeight + "px) Image to (" + resizedWidth + "px, " + resizedHeight + "px)")

    // draw image on an HTML canvas
    var canvas = document.createElement('canvas');
    canvas.width = resizedWidth;
    canvas.height = resizedHeight;
    canvas.getContext('2d').drawImage(img, 0, 0, resizedWidth, resizedHeight);

    // Get pixel data from a 224 x 224 crop of image 
    if (resizedHeight > resizedWidth) {
      // vertically centered crop works best for vertical images
      var data = canvas.getContext('2d').getImageData(0, (resizedHeight - 224)/2, 224, 224 + (resizedHeight - 224)/2).data;
    } else {
      // left crop works best for horizontal images
      var data = canvas.getContext('2d').getImageData(0, 0, 224, 224).data;
    }

    // Rearrange values to a 3 channel 224 by 224 matrix (ignoring a value given by context.getImageData) 
    const dataFromImage = ndarray(new Float32Array(data), [224, 224, 4]);
    const dataProcessed = ndarray(new Float32Array(224 * 224 * 3), [1, 3, 224, 224]);
    ops.assign(dataProcessed.pick(0, 0, null, null), dataFromImage.pick(null, null, 2));
    ops.assign(dataProcessed.pick(0, 1, null, null), dataFromImage.pick(null, null, 1));
    ops.assign(dataProcessed.pick(0, 2, null, null), dataFromImage.pick(null, null, 0));

    // Convert values from 0-255 to 0-1
    ops.divseq(dataProcessed, 255);

    // Normalize values so the mean and std for (r, g, b) is (0.485, 0.456, 0.406) and (0.229, 0.224, 0.225) respectively
    ops.subseq(dataProcessed.pick(0, 0, null, null), 0.485);
    ops.subseq(dataProcessed.pick(0, 1, null, null), 0.456);
    ops.subseq(dataProcessed.pick(0, 2, null, null), 0.406);
    ops.divseq(dataProcessed.pick(0, 0, null, null), 0.229);
    ops.divseq(dataProcessed.pick(0, 1, null, null), 0.224);
    ops.divseq(dataProcessed.pick(0, 2, null, null), 0.225);

    // Convert finalized data to ONNX Tensor
    var imgTensor = new Tensor((dataProcessed.data as Float32Array), 'float32', [1, 3, 224, 224])

    return imgTensor;
  }

  classifyImage = async () => {
    var inputImg = document.getElementById('imageView') as HTMLImageElement;

    // Load the ResNet18 Client ONNX model on first inference only
    if (!this.state.modelLoaded) {
      this.setState({
        loading: "Loading client model..."
      });
      await this.session.loadModel(RESNET18_CLIENT_MODEL_URL);
      console.log("loaded client model from " + RESNET18_CLIENT_MODEL_URL)
      this.setState({
        modelLoaded: true
      });
    }

    if (inputImg) {
      // Preprocess Image and perform client model inference
      this.setState({
        loading: "Client model inference..."
      });
      var imgTensor = this.preprocessImage(inputImg);
      console.log("Input Image Tensor: " + imgTensor.data)
      const clientOutputMap = await this.session.run([imgTensor])
      const splitActivation = clientOutputMap.values().next().value;
      console.log("Output of Client Model (splitActivation): " + splitActivation.data)

      // Perform server model inference on the split activation
      this.setState({
        loading: "Sending split activations for server model inference..."
      });
      const requestOptions = {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({dims: splitActivation.dims, data: Array.from(splitActivation.data)})
      };
      fetch(RESNET18_SERVER_MODEL_URL, requestOptions)
          .then(response => response.json())
          .then(data => {
            // Display predicted class through app state
            this.setState({class: data.class, loading: null});
          });
    } else {
      console.error("There was an error accessing the uploaded image. Please try again.")
    }
  }

  /**
   * Displays an image preview to the user whenever they upload an image
   */
  updatePreview = () => {
    document.getElementById("imageLoader").style.display = "block";
    const file = (document.getElementById("inputImage") as HTMLInputElement).files[0];
    console.log(file)
    if (file) {
        this.setState({ imageURL: URL.createObjectURL(file) });
    }
    document.getElementById("imageLoader").style.display = "none";
  }

  render() {
    return (
      <div className="App">
        <header className="App-header">
          <h1>ResNet18 Split Inference Demo</h1>
          <p className="desc">Your uploaded image is sent through the client model which returns the split activations of a cut layer within the full ResNet18 model. These activations are sent to a server where the remaining of the inference is completed. This allows you to perform inference on a model present on a third party server while preserving your image's privacy.</p>
          { (this.state.loading === null) ? (
            <div>
              <input onChange={this.updatePreview} type="file" id="inputImage" accept="image/jpeg" />
              <br />
              <input type="submit" onClick={this.classifyImage} value="Classify Image" id="submit" />
              <div id="imageLoader" className="loader"></div>
              { this.state.class ? <p>Class: {this.state.class}</p> : <span><br /><br /></span> }
              <img id="imageView" src={this.state.imageURL}/>
            </div>
          ) : (
            <p>{this.state.loading}</p>
          ) }
        </header>
      </div>
    )
  }

}

export default App;
