import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;

/**
 * Created by agibsonccc on 9/16/15.
 * Adapted by Erik Babu on 3/14/17
 * Notes by Erik Babu: In this class, the majority of the code has not been
 * written by me. However, I have annotated most of the lines of code, and it
 * should make sense how it works if you have read all the contents of our
 * website. I have also changed some of the variables to see if I can
 * improve the accuracy of the image classification. This class trains the
 * network
 * Note: This must be run before trying the TestYourOwnImage class.
 */
public class LeNetCNN {
  private static final Logger log = LoggerFactory.getLogger(LeNetCNN.class);

  public static void main(String[] args) throws Exception {
    // Number of input channels
    int nChannels = 1;

    // The number of possible outcomes. In this case, possible outcomes are the
    // digits 0 - 9
    int outputNum = 10;

    // Number of training examples in one forward/backward pass
    int batchSize = 64;

    // Number of forward and backward passes of all the training examples
    //1 forward pass and 1 backward pass is equivalent to 1 Epoch
    int nEpochs = 1;

    // Number of passes. Every pass uses n examples where n = batchSize
    int iterations = 1;

    //random number seed for reproducibility
    int seed = 123;


    //Create an iterator using the batch size for one iteration
    //Network to be trained using MNIST DataSet
    log.info("Load data....");
    DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, 12345);
    DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, 12345);

    //Construct the neural network
    log.info("Constructing neural network....");
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .seed(seed)
        .iterations(iterations) // Training iterations as above
        //regularization prevents overfitting
        .regularization(true).l2(0.0005)
        //set learning rate here. If the value is too low, it could take a
        //long time to find the global minimum. If it is too high, it may lead
        // to fluctuation or even divergence from the minimum
        .learningRate(.01)
        //Xavier algorithm automatically determines scale of initialization
        //it does this based on number of input and output neurons.
        .weightInit(WeightInit.XAVIER)
        //Type of gradient descent used
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        //Nesterovs algorithm used for accelerated gradient descent
        .updater(Updater.NESTEROVS).momentum(0.9)
        .list()
        //Pass input through first layer : Convolutional Layer
        //Set kernel size: 5 x 5 matrix used here
        .layer(0, new ConvolutionLayer.Builder(5, 5)
            //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
            .nIn(nChannels)
            .stride(1, 1)
            .nOut(20)
            .activation(Activation.IDENTITY)
            .build())
        //Pass output from conv layer through pooling layer
        //max pooling used here
        .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
            .kernelSize(2, 2)
            .stride(2, 2)
            .build())
        //Pass output from pooling layer through another layer of convolution
        .layer(2, new ConvolutionLayer.Builder(5, 5)
            //Note that nIn need not be specified in later layers
            .stride(1, 1)
            .nOut(50)
            .activation(Activation.IDENTITY)
            .build())
        .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
            .kernelSize(2, 2)
            .stride(2, 2)
            .build())
        //Introduce non-linearity by passing input through an activation
        // function. ReLU used in this case.
        .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
            .nOut(500).build())
        //Fully connected layer
        //Softmax activation function used here to guarantee sum of all
        // probabilities equals 1
        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .nOut(outputNum)
            .activation(Activation.SOFTMAX)
            .build())
        //Defines input to be image with dimensions of 28 * 28 pixels with 1
        // channel. Input is fed as a 1D array of 784 or 28 * 28 pixels.
        .setInputType(InputType.convolutionalFlat(28, 28, 1)) //See note below
        .backprop(true).pretrain(false).build();

    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();

    //Train the network
    log.info("Train model....");
    model.setListeners(new ScoreIterationListener(1));
    for (int i = 0; i < nEpochs; i++) {
      model.fit(mnistTrain);
      log.info("*** Completed epoch {} ***", i);

      //Evaluate the network after each iteration of backprop
      log.info("Evaluate network....");
      Evaluation eval = new Evaluation(outputNum);
      //Counter used just to minimise time spent waiting for data to be trained
      //Accuracy should still be relatively high after 2000 iterations
      while (mnistTest.hasNext()) {
        DataSet ds = mnistTest.next();
        //get the networks prediction
        INDArray output = model.output(ds.getFeatureMatrix(), false);
        //compare prediction to labelled output
        eval.eval(ds.getLabels(), output);
      }
      log.info(eval.stats());
      mnistTest.reset();
    }
    //Accuracy - The percentage of MNIST images that were correctly identified by our model.
    //Precision - The number of true positives divided by the number of true
      //positives and false positives.
    //Recall - The number of true positives divided by the number of true
      //positives and the number of false negatives.
    //F1 Score - Weighted average of precision and recall.
    System.out.println("TRAINING COMPLETE");

    //Save the model
    File locationToSave = new File("MyCNN.zip");
    boolean saveUpdater = true;
    ModelSerializer.writeModel(model, locationToSave, saveUpdater);

  }
}
