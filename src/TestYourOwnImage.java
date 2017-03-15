import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import javax.swing.*;
import java.io.File;
import java.util.Arrays;
import java.util.List;


/**
 * /**
 Written by Erik Babu:
 For this to work, you must first train the network by running the LeNetCNN
 class. The Neural Net that has been trained will then be used to classify
 the image you input. To input image, run the class, locate the image you
 would like to test and click the open button. After a short period of time,
 you should get output with the list of probabilities as well as the
 network's prediction
 */
public class TestYourOwnImage {
  private static Logger log = LoggerFactory.getLogger(TestYourOwnImage.class);


  /*
  Create a popup window to allow you to chose an image file to test against the
  trained Neural Network
  Chosen images will be automatically
  scaled to 28*28 grayscale
   */
  public static String fileChose() {
    JFileChooser fc = new JFileChooser();
    int ret = fc.showOpenDialog(null);
    if (ret == JFileChooser.APPROVE_OPTION) {
      File file = fc.getSelectedFile();
      String filename = file.getAbsolutePath();
      return filename;
    } else {
      return null;
    }
  }

  public static void main(String[] args) throws Exception {
    int height = 28;
    int width = 28;
    int channels = 1;

    List<Integer> labelList = Arrays.asList(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);

    // pop up file chooser
    String filechose = fileChose().toString();


    // Location of saved model
    File locationToSave = new File("MyCNN.zip");
    // Check for presence of saved model
    if (locationToSave.exists()) {
      System.out.println("\n######Saved Model Found######\n");
    } else {
      System.out.println("\n\n#######File not found!#######");
      System.out.println("Run LeNetCNN first to train data");
      System.out.println("#############################\n\n");

      System.exit(0);
    }

    //Load model
    MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork
        (locationToSave);

    log.info("*********TEST YOUR IMAGE AGAINST SAVED NETWORK********");

    // FileChose is a string we will need a file
    File file = new File(filechose);

    // Use NativeImageLoader to convert to numerical matrix
    NativeImageLoader loader = new NativeImageLoader(height, width, channels);

    // Get the image into an INDarray
    INDArray image = loader.asMatrix(file);

    // 0-255
    // 0-1
    DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    scaler.transform(image);

    INDArray output = model.output(image);

    System.out.println("File chosen was: " + filechose);

    //Function written to output the prediction
    int prediction = 0;
    float predictionVal = 0;
    float[] probabilities = new float[10];
    for (int i = 0; i <= 9; i++) {
      probabilities[i] = output.getFloat(0, i);
    }
    for (int i = 0; i <= 9; i++) {
      if (probabilities[i] > predictionVal) {
        prediction = i;
        predictionVal = probabilities[i];
      }
    }

    System.out.println("List of probabilities per label");
    System.out.println(output.toString());
    System.out.println((labelList.toString()));
    System.out.println();
    System.out.println("The Neural Net predicts, based on the training it " +
        "received in the LeNetCNN class : " + prediction);
  }

}
