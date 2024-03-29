/*
 * Author: Varun Thvar
 * Date of Creation: 21 February 2023
 * Description: This is an implementation of a simple trainable A-B-C Network
 *              designed for OR, AND, and XOR, all at the same time. This network incorporates the
 *              gradient descent algorithm to minimize the error by adjusting the weights by the
 *              derivative of the Error function.
 */

#include <cfloat>
#include <cmath>
#include <iostream>
#include <time.h>
#include <fstream>


#define NUMBER_ACTIVATIONS       ((int) 2)
#define NUMBER_HIDDEN_NODES      ((int) 5)
#define NUMBER_OUTPUTS           ((int) 3)
#define NUMBER_HIDDEN_LAYERS     ((int) 1)
#define NUM_CASES                ((int) 4)

#define MAX_ITERATIONS           ((int) 100000)
#define LAMBDA                   ((double) 0.3)
#define RANDOM_MIN               ((double) 0.1)
#define RANDOM_MAX               ((double) 0.5)
#define ERROR_THRESHOLD          ((double) (2.0e-4))

#define TRAIN                    ((bool) true)
#define RANDOMIZE                ((bool) true)
#define WEIGHTS_FILE             ((string) "../OptionI")
#define TRAINING_FILE            "../TrainingData.txt"
#define TRAINING_ANSWERS         "../TrainingAnswers.txt"
#define TESTING_FILE             "../TestData.txt"
#define SAVE_WEIGHTS             ((bool) false)

#define MILLISECONDS_IN_SECOND   ((double) 1000.0)
#define SECONDS_IN_MINUTE        ((double) 60.0)
#define MINUTES_IN_HOUR          ((double) 60.0)
#define HOURS_IN_DAY             ((double) 24.0)
#define DAYS_IN_WEEK             ((double) 7.0)


using namespace std;

/**
 * This implements the sigmoid function, which is defined by
 *    sigmoid(x) = 1/(1 + e^(-x))
 */
double sigmoid(double value)
{
   return 1.0/(1.0 + exp(-value));
} // double sigmoid(double value)

/**
 * This implements the derivative of the sigmoid function, which is defined by
 *    (d/dx) sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)), where sigmoid is defined by
 *    sigmoid(x) = 1/(1 + e^(-x))
 */
double sigmoidPrime(double value)
{
    double sig = sigmoid(value);
    return sig * (1.0 - sig);
} // double sigmoidPrime(double value)

/**
 * Accept a value representing seconds elapsed and print out a decimal value in easier to digest units
 * The magic numbers 1000., 60., 60., 24. and 7. need to be replaced with constants.
 *
 * Written by Dr. Eric Nelson. Modified by Varun Thvar
 */
void printTime(double seconds)
{
   double minutes, hours, days, weeks;

   if (seconds < 1.)
      printf("%g milliseconds", seconds * MILLISECONDS_IN_SECOND);
   else if (seconds < SECONDS_IN_MINUTE)
      printf("%g seconds", seconds);
   else
   {
      minutes = seconds / SECONDS_IN_MINUTE;

      if (minutes < MINUTES_IN_HOUR)
         printf("%g minutes", minutes);
      else
      {
         hours = minutes / MINUTES_IN_HOUR;

         if (hours < HOURS_IN_DAY)
            printf("%g hours", hours);
         else
         {
            days = hours / HOURS_IN_DAY;

            if (days < DAYS_IN_WEEK)
               printf("%g days", days);
            else
            {
               weeks = days / DAYS_IN_WEEK;

               printf("%g weeks", weeks);
            }
         } // if (hours < 24.)...else
      } // if (minutes < 60.)...else
   } // else if (seconds < 60.)...else
   cout << endl;
   return;
} // void printTime(double seconds)

/**
 * This is the structure of the Neural Network, which contains the following methods:
 *   randomValue() - Outputs a random value based on the range as given in the configuration parameters
 *   setConfigurationParameters() - Sets the configuration parameters for the network based on the following parameters
 *   echoConfigurationParameters() - Outputs the configuration parameters for the network
 *   allocateArrayMemory() - Allocates memory for the activations, hidden nodes, and weights of the network
 *   populateArrays() - Populates the arrays with random values, unless the network is a 2-2-1 network, in which
 *          it manually overrides
 *   checkNetwork() - Outputs the Network Type, Lambda Value, Error Threshold, Maximum Number of Iterations,
 *          and the Random Number Range
 *   train() - Trains the network using predetermined training data. The training is done using the gradient
 *          descent algorithm, which is used to minimize the error by adjusting the weights derivative of the error
 *          with respect to the weights
 *   run() - Runs the network using predetermined test data. Each node is calculated using the sigmoid function
 *          applied onto a dot product of the weights and the activations
 *   runTrain() - Runs the network ONLY DURING TRAINING using predetermined test data. Each node is calculated using the
 *          sigmoid function applied onto a dot product of the weights and the activations.
 *   printArray() - Prints array in a sequence for the purposes of reporting results after running
 *   reportResults() - Reports the results of the training or running of the network, depending on the mode
 *          the network is in training mode or not
 *
 */
struct NeuralNetwork
{
   int numOutputActivations;     // Number of Output Activations
   int numInputActivations;      // Number of Input Activations
   int numHiddenActivations;     // Number of Hidden Nodes in each Hidden Layer
   int numHiddenLayers;          // Number of Hidden Layers
   double lambda;                // Learning Rate - changes how much effect the derivative of the error has on the weights
   double errorThreshold;        // Threshold for the error to reach during training
   int maxIterations;            // Maximum number of iterations during training
   double randMin;               // Minimum value of the random value assigned to weights
   double randMax;               // Maximum value of the random value assigned to weights
   bool training;                // Whether or not the network is in training mode (the alternative being running mode)

   double* a;                    // Array for Input Activations
   double* h;                    // Array for Hidden Activations
   double* F;                    // Output Value
   double** weightsIJ;           // Weights between the Input Layer and the Hidden Layers
   double** weightsJK;           // Weights between the Hidden Layers and the Output Layer
   double** trainData;           // Training Data (Inputs)
   double** trainAnswers;        // TrainingAnswers.txt (Expected Outputs)
   double* thetaJ;               // Values used for calculating the hidden nodes - dot product of activations and  weights
   double* thetaI;               // Values used for calculating the Output - dot product of hidden layers and corresponding weights
   double** testData;            // Test Data (Inputs)
   double** deltaWIJ;            // Value of -1 * lambda * EPrimeIJ -> indicates the change in weightsIJ
   double** deltaWJK;            // Value of -1 * lambda * EPrimeJK -> indicates the change in weightsJK
   double* capitalOmega;         // Values of lowerPsi * weightsIJ[J]
   double* capitalPsi;           // Values of capitalOmega * sigmoidPrime(thetaJ)

   int numCases;                 // Number of Test Cases
   double trainingTime;          // Time taken for training the network
   double runningTime;           // Time taken for running the network
   bool randomize;               // Whether or not the network is in randomize mode
   int iterations;               // Number of iterations taken during training
   double errorReached;          // Error value reached at the end of training or running
   string reasonEndTraining;     // Reason for ending training
   bool savingWeights;             // True if the network  saves the weights at the end of training, false if otherwise
   string fileWeights;       // File path for loading the weights instead of randomizing

   /**
    * Outputs a random value based on the range as given in the configuration parameters
    */
   double randomValue()
   {
      return ((double) rand() / (RAND_MAX)) * (randMax - randMin) + randMin;
   } // double randomValue()

   /**
    * Sets the configuration parameters for the network
    */
   void setConfigurationParameters()
   {
      numOutputActivations = NUMBER_OUTPUTS;
      numCases = NUM_CASES;

      numInputActivations = NUMBER_ACTIVATIONS;
      numHiddenLayers = NUMBER_HIDDEN_LAYERS;
      numHiddenActivations = NUMBER_HIDDEN_NODES;

      lambda = LAMBDA;
      errorThreshold = ERROR_THRESHOLD;
      maxIterations = MAX_ITERATIONS;
      randMin = RANDOM_MIN;
      randMax = RANDOM_MAX;

      training = TRAIN;
      randomize = RANDOMIZE;
      savingWeights = SAVE_WEIGHTS;
      fileWeights = WEIGHTS_FILE;

      return;
   } // void setConfigurationParameters(int numAct, int numHidLayer ...

   /**
    * Outputs the configuration parameters for the network to check if they are set correctly
    */
   void echoConfigurationParameters()
   {
      cout << endl << "Echoing Configuration Parameters:" << endl;
      cout << "Network Type: " << numInputActivations << "-" << numHiddenActivations << "-" <<
         numOutputActivations << endl;
      if (training)
      {
         cout << "Lambda Value: " << lambda << endl;
         cout << "Error Threshold: " << errorThreshold << endl;
         cout << "Maximum Number of Iterations: " << maxIterations << endl;
         cout << "Random Value Minimum: " << randMin << endl;
         cout << "Random Value Maximum: " << randMax << endl;
         cout << "Randomizing Weights: " << (randomize ? "True" : "False") << endl;
      }
      return;
   } //void echoConfigurationParameters()

   /**
    * IF RUNNING: Allocates memory for the activations, hidden nodes, and weights of the network
    * IF TRAINING: Allocates memory for the activations, hidden nodes, weights, delta weights,
    *              training data, and test data of the network
    */
   void allocateArrayMemory()
   {
      int I, J, K, D;

      a = new double[numInputActivations]; // Initializing input and hidden activations
      h = new double[numHiddenActivations];
      F = new double[numOutputActivations];

      weightsIJ = new double*[numOutputActivations]; // Initializing weights
      for (I = 0; I < numOutputActivations; ++I) weightsIJ[I] = new double[numHiddenActivations];
      weightsJK = new double*[numHiddenActivations];
      for (J = 0; J < numHiddenActivations; ++J) weightsJK[J] = new double[numInputActivations];

      testData = new double*[numCases]; // Initializing Test Data
      for (int index = 0; index < numCases; ++index) testData[index] = new double[numInputActivations];

      if (training)
      {
         capitalOmega = new double[numHiddenActivations]; // Initializing capital Omega and Psi
         capitalPsi = new double[numHiddenActivations];

         thetaI = new double[numOutputActivations]; // Initializing Thetas
         thetaJ = new double[numHiddenActivations];

         deltaWIJ = new double*[numOutputActivations]; // Initializing delta weights
         for (I = 0; I < numOutputActivations; ++I) deltaWIJ[I] = new double[numHiddenActivations];
         deltaWJK = new double*[numHiddenActivations];
         for (J = 0; J < numHiddenActivations; ++J) deltaWJK[J] = new double[numInputActivations];

         trainData = new double*[numCases]; // Initializing Training Data
         for (int index = 0; index < numCases; ++index) trainData[index] = new double[numInputActivations];
         trainAnswers = new double*[numCases];
         for (int index = 0; index < numCases; ++index) trainAnswers[index] = new double[numOutputActivations];
      } // if (training)

      cout << "Allocated Memory!" << endl;
      return;
   } //void allocateArrayMemory()

   /**
    * IF RUNNING: Populates the weights with random values, unless the network is not in randomize mode in which
    *             it manually loads the weights. All other arrays (inputs, hiddens, output) are auto set to 0.0.
    *
    * IF TRAINING: Populates the weights with random values, unless the network is not in randomize mode in which
    *              it manually loads the weights. All other arrays (inputs, hiddens, output, thetas) are auto set to
    *              0.0.
    */
   void populateArrays()
   {
      int I, J, K;

      if (randomize) // Randomizing Weights
      {
         for (I = 0; I < numOutputActivations; ++I)
            for (int J = 0; J < numHiddenActivations; ++J)
               weightsIJ[I][J] = randomValue();
         for (J = 0; J < numHiddenActivations; ++J)
            for (int K = 0; K < numInputActivations; ++K)
               weightsJK[J][K] = randomValue();
      } // if (randomize)

      if (!randomize) // Loading Weights
      {
         ifstream file;
         file.open(fileWeights);
         string configuration = to_string(numInputActivations) + "_" + to_string(numHiddenActivations)
                           + "_" + to_string(numOutputActivations);

         if (file.is_open())
         {
            string firstLine;
            getline(file, firstLine);
            if (firstLine == configuration)
            {
               string valueString;
               double value;

               for (K = 0; K < numInputActivations; K++)
               {
                  for (J = 0; J < numHiddenActivations; J++)
                  {
                     getline(file, valueString);
                     value = stod(valueString);
                     weightsJK[J][K] = value;
                  } // for (J = 0; J < numHiddenActivations; J++)
               } // for (K = 0; K < numInputActivations; K++)

               for (J = 0; J < numHiddenActivations; J++)
               {
                  for (I = 0; I < numOutputActivations; I++)
                  {
                     getline(file, valueString);
                     value = stod(valueString);
                     weightsIJ[I][J] = value;
                  } // for (I = 0; I < numOutputActivations; I++)
               } // for (J = 0; J < numHiddenActivations; J++)
            } // if (firstLine == configuration)
            else // (firstLine != configuration)
            {
               cout << "File configuration does not match - all weights are set to default" << endl;
            } // else // (firstLine != configuration)
            file.close();
         } // if (file.is_open())

         else // (!file.is_open())
         {
            cout << "File not loaded - all weights are set to default" << endl;
         } // else (!file.is_open())
      } // if (!randomize)

      testData[0][0] = 0.0; // Initializing Testing Data
      testData[0][1] = 0.0;
      testData[1][0] = 0.0;
      testData[1][1] = 1.0;
      testData[2][0] = 1.0;
      testData[2][1] = 0.0;
      testData[3][0] = 1.0;
      testData[3][1] = 1.0;

      if (training)
      {
         trainData = testData;

         trainAnswers[0][0] = 0.0;
         trainAnswers[0][1] = 0.0;
         trainAnswers[0][2] = 0.0;

         trainAnswers[1][0] = 0.0;
         trainAnswers[1][1] = 1.0;
         trainAnswers[1][2] = 1.0;

         trainAnswers[2][0] = 0.0;
         trainAnswers[2][1] = 1.0;
         trainAnswers[2][2] = 1.0;

         trainAnswers[3][0] = 1.0;
         trainAnswers[3][1] = 1.0;
         trainAnswers[3][2] = 0.0;

      } // if (training)

      cout << "Populated Arrays!" << endl;
      return;
   } //void populateArrays()

   /**
    * Outputs the Network Type, Lambda Value, Error Threshold, Maximum Number of Iterations,
    *    and the Random Number Range. To be used before training and/or running.
    */
   void checkNetwork()
   {
      if (!training)
         cout << "Network Type: " << numInputActivations << "-" << numHiddenActivations << "-"
         << numOutputActivations << endl;
      if (training)
      {
         cout << "Training the Network!" << endl;
         cout << "Network Type: " << numInputActivations << "-" << numHiddenActivations << "-"
              << numOutputActivations << endl;
         cout << "Lambda Value: " << lambda << endl;
         cout << "Error Threshold: " << errorThreshold << endl;
         cout << "Maximum Number of Iterations: " << maxIterations << endl;
         cout << "Random Value Minimum: " << randMin << endl;
         cout << "Random Value Maximum: " << randMax << endl;
         cout << endl;
      } // if (training)
      return;
   } //void checkNetwork()

   /**
   * Runs the network using predetermined test data. Used for solely running purposes.
   *     Each node is calculated using the sigmoid function applied onto a dot product
   *     of the weights and the activations.
    */
   double* run(double *inputValues)
   {
      time_t dummyStart, dummyEnd;
      int I, J, K;
      double thetaJ;
      double thetaI;

      time(&dummyStart);

      a = inputValues;

      for (J = 0; J < numHiddenActivations; ++J)
      {
         thetaJ = 0.0;
         for (K = 0; K < numInputActivations; ++K)
         {
            thetaJ += a[K] * weightsJK[J][K];
         } // for (K = 0; K < numInputActivations; ++K)
         h[J] = sigmoid(thetaJ);
      } // for (J = 0; J < numHiddenActivations; ++J)

      for (I = 0; I < numOutputActivations; ++I)
      {
         thetaI = 0.0;
         for (J = 0; J < numHiddenActivations; ++J)
         {
            thetaI += h[J] * weightsIJ[I][J];
         } // for (J = 0; J < numHiddenActivations; ++J)
         F[I] = sigmoid(thetaI);
      } // for (I = 0; I < numOutputActivations; ++I)

      time(&dummyEnd);
      runningTime = double(dummyEnd - dummyStart);
      return F;
   } // double run(double *inputValues)

   /**
    * Runs the network ONLY DURING TRAINING using predetermined test data. Each node is calculated using
    *    the sigmoid function applied onto a dot product of the weights and the activations.
    */
   double* runTrain(double *inputValues)
   {
      int I, J, K;
      a = inputValues;

      for (J = 0; J < numHiddenActivations; ++J)
      {
         thetaJ[J] = 0.0;
         for (K = 0; K < numInputActivations; ++K)
         {
            thetaJ[J] += a[K] * weightsJK[J][K];
         } // for (K = 0; K < numInputActivations; ++K)
         h[J] = sigmoid(thetaJ[J]);
      } // for (J = 0; J < numHiddenActivations; ++J)

      for (I = 0; I < numOutputActivations; ++I)
      {
         thetaI[I] = 0.0;
         for (J = 0; J < numHiddenActivations; ++J)
         {
            thetaI[I] += h[J] * weightsIJ[I][J];
         } // for (J = 0; J < numHiddenActivations; ++J)
         F[I] = sigmoid(thetaI[I]);
      } // for (I = 0; I < numOutputActivations; ++I)

      return F;
   } // double runTrain(double *inputValues)

   /**
    * Trains the network using predetermined training data using the gradient descent algorithm, which is used to
    *    minimize the error by adjusting the weights derivative of the error with respect to the weights. Uses
    *    runTrain() to calculate the activations and outputs of the network.
    */
   void train()
   {
      time_t dummyStart, dummyEnd;

      int I, J, K, D, epoch;
      double EPrimeIJ, EPrimeJK;
      double *lowerOmega, *lowerPsi, *testingArray, *answersArray;
      lowerOmega = new double[numOutputActivations];
      lowerPsi = new double[numOutputActivations];

      time(&dummyStart);
      checkNetwork();

      errorReached = DBL_MAX;

      while (epoch < maxIterations && errorReached > errorThreshold)
      {
         errorReached = 0.0;
         for (D = 0; D < numCases; ++D)
         {
            testingArray = trainData[D];
            answersArray = trainAnswers[D];
            runTrain(testingArray);

            for (I = 0; I < numOutputActivations; ++I)
            {
               lowerOmega[I] = answersArray[I] - F[I];
               lowerPsi[I] = lowerOmega[I] * sigmoidPrime(thetaI[I]);

               for (J = 0; J < numHiddenActivations; ++J)
               {
                  EPrimeIJ = -h[J] * lowerPsi[I];
                  deltaWIJ[I][J] = -lambda * EPrimeIJ;
               } // for (J = 0; J < numHiddenActivations; ++J)
            } // for (I = 0; I < numOutputActivations; ++I)

            for (J = 0; J < numHiddenActivations; ++J)
            {
               capitalOmega[J] = 0;
               for (I = 0; I < numOutputActivations; ++I)
               {
                  capitalOmega[J] += lowerPsi[I] * weightsIJ[I][J];
               } // for (I = 0; I < numOutputActivations; ++I)
               capitalPsi[J] = capitalOmega[J] * sigmoidPrime(thetaJ[J]);

               for (K = 0; K < numInputActivations; ++K)
               {
                  EPrimeJK = -a[K] * capitalPsi[J];
                  deltaWJK[J][K] = -lambda * EPrimeJK;
               } // for (K = 0; K < numInputActivations; ++K)
            } // for (J = 0; J < numHiddenActivations; ++J)

            for (J = 0; J < numHiddenActivations; ++J)
            {
               for (I = 0; I < numOutputActivations; ++I)
               {
                  weightsIJ[I][J] += deltaWIJ[I][J];
               }
               for (K = 0; K < numInputActivations; ++K)
               {
                  weightsJK[J][K] += deltaWJK[J][K];
               }
            } // for (J = 0; J < numHiddenActivations; ++J)

            runTrain(testingArray);
            for (I = 0; I < numOutputActivations; ++I)
            {
               errorReached += 0.5 * (answersArray[I] - F[I]) * (answersArray[I] - F[I]);
            } // for (I = 0; I < numOutputActivations; ++I)
         } // for (D = 0; D ...

         errorReached /= numCases;
         ++epoch;
      } // while (epoch < maxIterations && errorReached > errorThreshold)

      iterations = epoch;

      if (iterations == maxIterations) reasonEndTraining = "Maximum Number of Iterations Reached";
      else reasonEndTraining = "Error Threshold Reached";

      if (savingWeights) // Saving weights
      {
         string fileName = fileWeights;
         ifstream file;
         file.open(fileName);

         while (file) // Making sure a new file is created
         {
            file.close();
            fileName += "I";
            file.open(fileName);
         } // while (file)

         ofstream outfile(fileName);

         outfile << to_string(numInputActivations) + "_" + to_string(numHiddenActivations)
                           + "_" + to_string(numOutputActivations) << endl;

         for (K = 0; K < numInputActivations; K++)
         {
            for (J = 0; J < numHiddenActivations; J++)
            {
               outfile << to_string(weightsJK[J][K]) << endl;
            } // for (J = 0; J < numHiddenActivations; J++)
         } // for (K = 0; K < numInputActivations; K++)

         for (J = 0; J < numHiddenActivations; J++)
         {
            for (I = 0; I < numOutputActivations; I++)
            {
               outfile << to_string(weightsIJ[I][J]) << endl;
            } // for (I = 0; I < numOutputActivations; I++)
         } // for (J = 0; J < numHiddenActivations; J++)

         outfile.close();
      } // if (savingWeights)

      trainingTime = double(time(&dummyEnd) - dummyStart);

      return;
   } // void train()

   /**
    * Prints array in a sequence for the purposes of reporting results after running. Follows the format
    *   value1, value2, value3, ... valueN
    */
   void printArray(double* arr, int length)
   {
      for (int index = 0; index < length; ++index)
      {
         cout << arr[index];
         if (index != length - 1) cout << ", ";
      } // for (int index = 0; index < length; ++index)
      return;
   } // void printArray(double* arr, int length)

   /**
    * Reports the results of the training or running of the network, depending on the mode the network
    *    is in training mode or not
    */
   void reportResults()
   {
      if (training)
      {
         cout << "Reason for Termination: " << reasonEndTraining << endl;
         cout << "Training Time Taken: ";
         printTime(trainingTime);
         cout << "Error Reached: " << errorReached << endl;
         cout << "Iterations reached: " << iterations << endl << endl;
         cout << "Truth Table and Expected Outputs" << endl;

         for (int index = 0; index < numCases; ++index)
         {
            printArray(trainData[index], numInputActivations);
            cout << " = ";
            printArray(trainAnswers[index], numOutputActivations);
            cout << " -> ";
            printArray(run(trainData[index]), numOutputActivations);
            cout << endl;
         } // for (int index = 0...

         cout << endl;
      } // if (training)
      return;
   } // reportResults()
}; // struct NeuralNetwork


/**
 * Accepts input from the user and runs the network using the input values. Specific to Boolean Algebra
 */
void testingData(NeuralNetwork* network)
{
   for (int index = 0; index < network->numCases; ++index)
   {
      network->checkNetwork();
      cout << "Running the Network with Test Data: ";
      network->printArray(network->testData[index], network->numInputActivations);
      cout << endl;
      network->printArray(network->run(network->testData[index]),  network->numOutputActivations);
      cout << endl << endl;
   } // for (int numOutputActivations = 0; numOutputActivations < 4; ++numOutputActivations)
   return;
} // void testingData(NeuralNetwork* network)

/**
 * Main function of the program - creates and configures the network, trains it, and then runs it for all test cases
 */
int main(int argc, char *argv[])
{
   srand(time(NULL));
   rand();

   NeuralNetwork network; // Creating and Configurating the Network based on pre-determined constants and designs
   network.setConfigurationParameters();
   network.echoConfigurationParameters();
   cout << endl;

   network.allocateArrayMemory(); // Allocating Arrays in Network
   network.populateArrays(); // Populating Arrays in Network
   cout << endl;

   if (network.training) // Training the Network using predetermined training data
   {
      network.train();
      network.reportResults();
   } // if (network.training)

   network.training = false; // Running the Network using test data
   testingData(&network);
   network.reportResults();

   return 0;
} // int main(int argc, char *argv[])




/**
 * This is the structure of the FileReader, which contains the following methods:
 *    FileReader(string file) - Constructor for the FileReader, which accepts a file path
 *    readData() - Reads the data from the file and populates the arrays in the network
 */
struct FileReader
{
   ifstream file;

   /**
    * Constructor for the FileReader, which accepts a file path and opens the file
    */
   FileReader(string fileName)
   {
      file.open(fileName);
      if (!file.is_open())
      {
         std::cerr << "Couldn't open config file for reading.\n";
      }
      return;
   } // FileReader(string fileName)

   /**
    * Reads the next line from the file and returns the value of the line after the "=" and before the ";"
    */
   string readNextLine()
   {
      string line;
      bool notAtLine = true;
      string value;

      while(notAtLine)
      {
         getline(file, line);
         if (!(line[0] == '_' || line.empty()))
         {
            notAtLine = false;
            auto delimiterPos = line.find('=');
            auto delimiterEnd = line.find(';');
            auto name = line.substr(0, delimiterPos);
            value = line.substr(delimiterPos + 1, delimiterEnd - delimiterPos - 1);
         } // if (!(line[0] == '_' || line.empty()))
      } // while(notAtLine)
      return value;
   } // string readNextLine()

   /**
    * Closes the file
    */
   void closeFile()
   {
      file.close();
      /*
      *numInputActivations = stoi(reader->readNextLine());
      numHiddenActivations = stoi(reader->readNextLine());
      numOutputActivations = stoi(reader->readNextLine());
      numCases = stoi(reader->readNextLine());
      maxIterations = stoi(reader->readNextLine());
      lambda = stod(reader->readNextLine());
      randMin = stod(reader->readNextLine());
      randMax = stod(reader->readNextLine());
      errorThreshold = stod(reader->readNextLine());
      training = stob(reader->readNextLine());
      randomize = stob(reader->readNextLine());
      savingWeights = stob(reader->readNextLine());
      fileWeights = (reader->readNextLine());
      trainingFile = (reader->readNextLine());
      trainingAnswersFile = (reader->readNextLine());
      testingFile = (reader->readNextLine());
      reader->closeFile();
      delete reader;
      */
      return;
   } // void closeFile()
}; // struct FileReader
