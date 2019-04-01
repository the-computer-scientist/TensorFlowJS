const tf = require("@tensorflow/tfjs-node");
const data = require("./tfjs-examples/mnist-node/data.js");

async function getData() {
  await data.loadData();
	[trainImages, trainLabels, testImages, testLabels] = data.dataset;
	trainImages = tf.tensor(trainImages).reshape([-1,28,28,1]);
	trainLabels = tf.tensor(trainLabels).oneHot(10).squeeze(1);
	testImages = tf.tensor(testImages).reshape([-1,28,28,1]);
	testLabels = tf.tensor(testLabels).oneHot(10).squeeze(1);
}

async function train() {
	const model = tf.sequential({layers: [
		tf.layers.conv2d({filters: 8, kernelSize: 3, activation: "relu", inputShape: [28,28,1]}),
		tf.layers.maxPooling2d({poolSize: [2,2], strides: [2,2]}),
		tf.layers.flatten(),
		tf.layers.dense({units: 10, activation: "softmax"})
	]});

	model.compile({optimizer: tf.train.adam(), loss: tf.losses.logLoss, metrics: ["accuracy"]});
	await model.fit(trainImages, trainLabels, {epochs: 5, batchSize: 50});
	await model.save("file://model/");
}

async function evaluate() {
	const model = await tf.loadLayersModel("file://model/model.json");
	model.summary();
	model.compile({optimizer: tf.train.adam(), loss: tf.losses.logLoss, metrics: ["accuracy"]});
	const result = model.evaluate(testImages, testLabels);
	console.log("Loss:", await result[0].data());
	console.log("Acc:", await result[1].data());
}

getData().then(train).then(evaluate);

function demo() {
	const m = tf.variable(tf.tensor(4.0));
	m.print();

	function f(x) {
		return tf.mul(m,x);
	}

	f(5).print();

	// (f(5) - 5)^2
	function loss() {
		return f(5).sub(5).square();
	}

	loss().print();

	const optimizer = tf.train.sgd(0.01);
	for(let i=0; i<20; i++) {
		optimizer.minimize(loss);
		console.log(m.dataSync(), f(5).dataSync());
	}
}