(this.webpackJsonpresnet_react_ts_app=this.webpackJsonpresnet_react_ts_app||[]).push([[0],{17:function(e,t,n){},25:function(e,t,n){},27:function(e,t,n){"use strict";n.r(t);var s=n(2),a=n.n(s),i=n(7),l=n.n(i),c=(n(17),n(4)),r=n.n(c),o=n(8),d=n(9),u=n(10),p=n(12),j=n(11),m=n(5),b=n(6),g=n.n(b),h=n(1),x=n.n(h),f=(n(25),n(0)),O="https://splitlearning.mit.edu/SplitLearningInference/client_models/",v=function(e){Object(p.a)(n,e);var t=Object(j.a)(n);function n(e){var s;return Object(d.a)(this,n),(s=t.call(this,e)).session=new m.InferenceSession({backendHint:"cpu"}),s.preprocessImage=function(e){var t=e.width,n=e.height,s=224,a=224;t>n?a=t/n*224:t<n&&(s=n/t*224),console.log("Resized ("+t+"px, "+n+"px) Image to ("+a+"px, "+s+"px)");var i=document.createElement("canvas");if(i.width=a,i.height=s,i.getContext("2d").drawImage(e,0,0,a,s),s>a)var l=i.getContext("2d").getImageData(0,(s-224)/2,224,224+(s-224)/2).data;else l=i.getContext("2d").getImageData(0,0,224,224).data;var c=g()(new Float32Array(l),[224,224,4]),r=g()(new Float32Array(150528),[1,3,224,224]);return x.a.assign(r.pick(0,0,null,null),c.pick(null,null,2)),x.a.assign(r.pick(0,1,null,null),c.pick(null,null,1)),x.a.assign(r.pick(0,2,null,null),c.pick(null,null,0)),x.a.divseq(r,255),x.a.subseq(r.pick(0,0,null,null),.485),x.a.subseq(r.pick(0,1,null,null),.456),x.a.subseq(r.pick(0,2,null,null),.406),x.a.divseq(r.pick(0,0,null,null),.229),x.a.divseq(r.pick(0,1,null,null),.224),x.a.divseq(r.pick(0,2,null,null),.225),new m.Tensor(r.data,"float32",[1,3,224,224])},s.classifyImage=Object(o.a)(r.a.mark((function e(){var t,n,a,i,l,c;return r.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:if(t=document.getElementById("imageView"),n=5,s.state.modelLoaded){e.next=8;break}return s.setState({loading:"Loading client model..."}),e.next=6,s.session.loadModel(O+"resnet18_client_sl_"+n+".onnx");case 6:console.log("loaded client model from "+O+"resnet18_client_sl_"+n+".onnx"),s.setState({modelLoaded:!0});case 8:if(!t){e.next=22;break}return s.setState({loading:"Client model inference..."}),a=s.preprocessImage(t),console.log("Input Image Tensor: "+a.data),e.next=14,s.session.run([a]);case 14:i=e.sent,l=i.values().next().value,console.log("Output of Client Model (splitActivation): "+l.data),s.setState({loading:"Sending split activations for server model inference..."}),c={method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({dims:l.dims,data:Array.from(l.data)})},fetch("https://sl-demo-backend.herokuapp.com/inference/"+n,c).then((function(e){return e.json()})).then((function(e){s.setState({class:e.class,loading:null})})),e.next=23;break;case 22:console.error("There was an error accessing the uploaded image. Please try again.");case 23:case"end":return e.stop()}}),e)}))),s.updatePreview=function(){document.getElementById("imageLoader").style.display="block";var e=document.getElementById("inputImage").files[0];console.log(e),e&&s.setState({imageURL:URL.createObjectURL(e)}),document.getElementById("imageLoader").style.display="none"},s.state={loading:null,modelLoaded:!1,class:null,imageURL:null},s}return Object(u.a)(n,[{key:"render",value:function(){return Object(f.jsx)("div",{className:"App",children:Object(f.jsxs)("header",{className:"App-header",children:[Object(f.jsx)("h1",{children:"ResNet18 Split Inference Demo"}),Object(f.jsxs)("p",{className:"desc",children:["A pre-trained ResNet18 model is split into ",Object(f.jsx)("b",{children:"two"})," separate models \u2014 ",Object(f.jsx)("b",{children:"client"})," (Alice) and ",Object(f.jsx)("b",{children:"server"})," (Bob)."]}),Object(f.jsx)("img",{id:"splitInferenceDiagram",src:"split_inference_diagram.png"}),Object(f.jsxs)("p",{className:"desc",children:["Your ",Object(f.jsx)("b",{children:"uploaded image"})," is sent through the client model which returns the ",Object(f.jsx)("b",{children:"split activations"})," of a cut layer within the full model.",Object(f.jsx)("br",{}),Object(f.jsx)("br",{}),"These activations are sent to a server where the remaining of the inference is completed. The ",Object(f.jsx)("b",{children:"output tensor"})," is then sent back to the client.",Object(f.jsx)("br",{}),Object(f.jsx)("br",{}),"This allows you to perform ",Object(f.jsx)("b",{children:"inference"})," on a model present on a ",Object(f.jsx)("b",{children:"third party"})," server while ",Object(f.jsx)("b",{children:"preserving your image's privacy"}),"."]}),null===this.state.loading?Object(f.jsxs)("div",{children:[Object(f.jsx)("input",{onChange:this.updatePreview,type:"file",id:"inputImage",accept:"image/jpeg"}),Object(f.jsx)("br",{}),Object(f.jsx)("br",{}),Object(f.jsx)("div",{id:"imageLoader",className:"loader"}),this.state.class?Object(f.jsxs)("p",{children:["Class: ",this.state.class]}):Object(f.jsxs)("span",{children:[Object(f.jsx)("br",{}),Object(f.jsx)("br",{})]}),Object(f.jsx)("img",{id:"imageView",src:this.state.imageURL,style:{display:null===this.state.imageURL?"none":"block"}}),Object(f.jsx)("input",{type:"submit",onClick:this.classifyImage,value:"Classify Image",id:"submit"})]}):Object(f.jsx)("p",{children:this.state.loading})]})})}}]),n}(s.Component),y=function(e){e&&e instanceof Function&&n.e(3).then(n.bind(null,28)).then((function(t){var n=t.getCLS,s=t.getFID,a=t.getFCP,i=t.getLCP,l=t.getTTFB;n(e),s(e),a(e),i(e),l(e)}))};l.a.render(Object(f.jsx)(a.a.StrictMode,{children:Object(f.jsx)(v,{})}),document.getElementById("root")),y()}},[[27,1,2]]]);
//# sourceMappingURL=main.2c28a9aa.chunk.js.map