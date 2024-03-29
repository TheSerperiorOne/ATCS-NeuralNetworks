/*
 * Author: Varun Thvar
 * Date of Creation: 25 March 2023
 * Description: This is an implementation of a simple trainable A-B-C-D Network
 *              designed for OR, AND, and XOR, all at the same time. This network incorporates the
 *              backprogation algorithm to minimize the error by adjusting the weights by the
 *              derivative of the Error function and has 2 hidden layers.
 *
 * TOML: This file uses the TOML++ Library written by Mark Gillard to parse the configuration file, a .toml file.
 *       The code to execute this via terminal is "g++ -I tomlplusplus/include -std=c++20 ABCDNetwork.cpp", and
 *       the user has to have the tomlplusplus library in the same directory as the file.
 */

#include <iostream>
#include <toml++/toml.hpp> // TOML Implementation from Toml++ Library

#define stringify(name)                    (# name)            // Converts a variable name to a string
#define MILLISECONDS_IN_SECOND             ((double) 1000.0)
#define SECONDS_IN_MINUTE                  ((double) 60.0)
#define MINUTES_IN_HOUR                    ((double) 60.0)
#define HOURS_IN_DAY                       ((double) 24.0)
#define DAYS_IN_WEEK                       ((double) 7.0)
#define DEFAULT_CONFIG_FILE                "config.toml"       // Has to be in the same directory as the executable
#define HIDDEN_LAYER_1                     1
#define HIDDEN_LAYER_2                     2

using namespace std;

string configFilePath = DEFAULT_CONFIG_FILE;
enum activationFunctionType {LINEAR, SIGMOID};

/**
 * Implements the linear function, which is defined by
 *    linear(x) = x
 */
double linear(double value)
{
   return value;
} // double linear(double value)

/**
 * Implements the derivative linear function, which is defined by
 *    linearPrime(x) = 1.0
 */
double linearPrime(double value)
{
   return 1.0;
} // double linearPrime(double value)

/**
 * This implements the sigmoid function, which is defined by
 *    sigmoid(x) = 1.0/(1.0 + e^(-x))
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
            } // else (days >= DAYS_IN_WEEK)
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
 *   loadWeights() - Loads the weights from the fileWeights file
 *   initializeTestingData() - Initializes the testing data based on the testing file
 *   initializeTrainingData() - Initializes the training data based on the training file
 *   initalizeTrainingAnswers() - Initializes the training answers based on the training answers file
 *   saveWeights() - Saves the weights to a file, which is called "WeightsI" + n x "I", where n is the number of files
 *       already created. It adds the configuration of the network to the first line of the file, and then adds the
 *       weights to the file.
 *   checkNetwork() - Outputs the Network Type, Lambda Value, Error Threshold, Maximum Number of Iterations,
 *          and the Random Number Range
 *   train() - Trains the network using training data. The training is done using the gradient
 *          descent algorithm, which is used to minimize the error by adjusting the weights derivative of the error
 *          with respect to the weights
 *   run() - Runs the network using  test data. Each node is calculated using the defined activation function
 *          applied onto a dot product of the weights and the activations
 *   runTrain() - Runs the network ONLY DURING TRAINING using predetermined test data. Each node is calculated
 *          using the specificed activation function applied onto a dot product of the weights and the activations.
 *   printArray() - Prints array in a sequence for the purposes of reporting results after running
 *   reportResults() - Reports the results of the training or running of the network, depending on the mode
 *          the network is in training mode or not
 *
 */
struct NeuralNetwork
{
/**
 * Configuration Parameters for the Network, such as the network configuration, number of cases, learning rate,
 *    error threshold, maximum number of iterations, random value range, and whether the network is in training mode.
 */
   int numOutputActivations;     // Number of Output Activations
   int numInputActivations;      // Number of Input Activations
   int* numHiddenActivations;    // Number of Hidden Nodes in each Hidden Layer
   int numHiddenLayers;          // Number of Hidden Layers
   double lambda;                // Learning Rate
   double errorThreshold;        // Threshold for the error to reach during training
   int maxIterations;            // Maximum number of iterations during training
   double randMin;               // Minimum value of the random value assigned to weights
   double randMax;               // Maximum value of the random value assigned to weights
   bool training;                // Whether or not the network is in training mode (true) or running mode
   int outputPrecision;          // Precision of the output values

/**
 * Activation Function and its Derivative, which are used to calculate the hidden nodes and the output nodes
 */
   double (*activationFunction) (double value);             // Activation Function
   double (*activationFunctionDerivative) (double value);   // Derivative of the Activation Function

/**
 * Arrays for the Network, including layers (a, h, F), weights, and training data,
 *    and backpropagation-relevant arrays such as thetaJ, and lowerPsi
 */
   double* a;                    // Array for Input Activations
   double** h;                   // Array for Hidden Activations
   double* F;                    // Output Value
   double** weightsIJ;           // Weights between the Input Layer and the Hidden Layers
   double** weightsJK;           // Weights between the first and second Hidden Layers
   double** weightsKM;           // Weights between the Hidden Layers and the Output Layer
   double* thetaJ;               // Values used to calculating the hidden nodes - dot product of inputs and weights
   double* thetaK;               // Values used to calculating the hidden nodes - dot product of inputs and weights
   double* lowerPsi;             // Value of lowerOmega multiplied by the derivative of the activation function function
   double** trainData;           // Training Data (Inputs)
   double** trainAnswers;        // TrainingAnswers.txt (Expected Outputs)
   double** testData;            // Test Data (Inputs)

/**
 * Relevant Information for Training and Running the Network, for the purposes of reporting results,
 *    or checking efficacy of the network
 */
   int numCases;                 // Number of Test Cases
   int trainingTime;             // Time taken for training the network
   double runningTime;           // Time taken for running the network
   int weightsConfiguration;     // 0 if the weights are randomized, 1 if loaded from file, 2 if manually set
   int iterations;               // Number of iterations taken during training
   double errorReached;          // Error value reached at the end of training or running
   bool savingWeights;           // True if the network  saves the weights at the end of training, false if otherwise
   string fileWeights;           // File path for loading the weights instead of randomizing
   string saveWeightsFile;       // File path for saving the weights
   string trainingFile;          // File path for the training data
   string trainingAnswersFile;   // File path for the training answers
   string testingFile;           // File path for the testing data

/**
 * Outputs a random value based on the range as given in the configuration parameters
 */
   double randomValue()
   {
      return ((double) rand() / (RAND_MAX)) * (randMax - randMin) + randMin;
   } // double randomValue()

/**
 * Sets the configuration parameters for the network using TOML++ Library
 */
   void setConfigurationParameters()
   {

      auto config = toml::parse_file(configFilePath);
      numInputActivations = *(config.get(stringify(numInputActivations))->value<int>());
      numOutputActivations = *(config.get(stringify(numOutputActivations))->value<int>());
      numHiddenLayers = *(config.get(stringify(numHiddenLayers))->value<int>());

      toml::array arr = *config.get_as<toml::array>("numHiddenActivations");
      numHiddenActivations = new int[numHiddenLayers];

      for (int index = 0; index < numHiddenLayers + 1; ++index)
      {
         numHiddenActivations[index] = *arr[index].value<int>();
      } // for (int index = 0; index < numHiddenLayers; ++index

      weightsConfiguration = *(config.get(stringify(weightsConfiguration))->value<int>());
      outputPrecision = *(config.get(stringify(outputPrecision))->value<int>());

      string activationFunctionString = *(config.get(stringify(activationFunction))->value<string>());
      if (activationFunctionString == stringify(linear))
      {
         activationFunction = linear;
         activationFunctionDerivative = linearPrime;
      } // if (activationFunctionString == "linear")
      else if (activationFunctionString == stringify(sigmoid))
      {
         activationFunction = sigmoid;
         activationFunctionDerivative = sigmoidPrime;
      } // else if (activationFunctionString == "sigmoid")

      training = *(config.get(stringify(training))->value<bool>());
      numCases = *(config.get(stringify(numCases))->value<int>());
      maxIterations = *(config.get(stringify(maxIterations))->value<int>());
      lambda = *(config.get(stringify(lambda))->value<double>());
      randMin = *(config.get(stringify(randMin))->value<double>());
      randMax = *(config.get(stringify(randMax))->value<double>());
      errorThreshold = *(config.get(stringify(errorThreshold))->value<double>());
      savingWeights = *(config.get(stringify(savingWeights))->value<bool>());

      fileWeights = *(config.get(stringify(fileWeights))->value<string>());
      saveWeightsFile = *(config.get(stringify(saveWeightsFile))->value<string>());
      trainingFile = *(config.get(stringify(trainingFile))->value<string>());
      trainingAnswersFile = *(config.get(stringify(trainingAnswersFile))->value<string>());
      testingFile = *(config.get(stringify(testingFile))->value<string>());
      return;
   } // void setConfigurationParameters()

/**
 * Outputs the configuration parameters for the network to check if they are set correctly
 */
   void echoConfigurationParameters()
   {
      cout << endl << "Echoing Configuration Parameters:" << endl;
      cout << "   Network Type: " << to_string(numInputActivations) << "-" << to_string(numHiddenActivations[HIDDEN_LAYER_1]) << "-" <<
         to_string(numHiddenActivations[HIDDEN_LAYER_2]) << "-" << to_string(numOutputActivations) << endl;
      cout << "   Input File: " << testingFile << endl;
      cout << "   Output Precision: " << outputPrecision << endl;
      cout << "   Activation Function: " << (activationFunction == linear ? "Linear" : "Sigmoid") << endl;
      if (weightsConfiguration == 1) cout << "Weights File: " << fileWeights << endl;
      if (training)
      {
         cout << "   Training Parameters:" << endl;
         cout << "     " << "Lambda Value: " << lambda << endl;
         cout << "     " << "Error Threshold: " << errorThreshold << endl;
         cout << "     " << "Maximum Number of Iterations: " << maxIterations << endl;
         cout << "     " << "Random Value Minimum: " << randMin << endl;
         cout << "     " << "Random Value Maximum: " << randMax << endl;
         cout << "     " << "Randomizing Weights: " << (weightsConfiguration == 0 ? "Randomize" :
            weightsConfiguration == 1 ? "Load from File" : "Manually Set") << endl;
         cout << "     " << "Saving Weights: " << (savingWeights ? "True": "False") << endl;
         cout << "     " << "Training File: " << trainingFile << endl;
         cout << "     " << "Training Answers File: " << trainingAnswersFile << endl;
         if (savingWeights) cout << "     " << "Weights Saved to: " << fileWeights << endl;
      } // if (training)
      return;
   } //void echoConfigurationParameters()

/**
 * IF RUNNING: Allocates memory for the activations, hidden nodes, weights, and test data of the network
 * IF TRAINING: Allocates memory for the activations, hidden nodes, weights, thetaJ, lower Psi,
 *     training data, and test data of the network
 */
   void allocateArrayMemory()
   {
      int I, J, K;

      a = new double[numInputActivations]; // Initializing input and hidden activations

      h = new double*[numHiddenLayers + 1];
      h[HIDDEN_LAYER_1] = new double[numHiddenActivations[HIDDEN_LAYER_1]];
      h[HIDDEN_LAYER_2] = new double[numHiddenActivations[HIDDEN_LAYER_2]];

      F = new double[numOutputActivations];

      weightsIJ = new double*[numOutputActivations]; // Initializing weights
      for (I = 0; I < numOutputActivations; ++I) weightsIJ[I] = new double[numHiddenActivations[HIDDEN_LAYER_2]];
      weightsJK = new double*[numHiddenActivations[HIDDEN_LAYER_2]];
      for (J = 0; J < numHiddenActivations[HIDDEN_LAYER_2]; ++J) weightsJK[J] = new double[numHiddenActivations[HIDDEN_LAYER_1]];
      weightsKM = new double*[numHiddenActivations[HIDDEN_LAYER_1]];
      for (K = 0; K < numHiddenActivations[HIDDEN_LAYER_1]; ++K) weightsKM[K] = new double[numInputActivations];

      testData = new double*[numCases]; // Initializing Test Data
      for (int index = 0; index < numCases; ++index) testData[index] = new double[numInputActivations];

      if (training) // Initializing Training Data and Training Answers and ThetaJ and LowerPsi
      {
         thetaJ = new double[numHiddenActivations[HIDDEN_LAYER_2]];
         thetaK = new double[numHiddenActivations[HIDDEN_LAYER_1]];
         lowerPsi = new double[numOutputActivations];

         trainData = new double*[numCases]; // Initializing Training Data
         for (int index = 0; index < numCases; ++index) trainData[index] = new double[numInputActivations];
         trainAnswers = new double*[numCases];
         for (int index = 0; index < numCases; ++index) trainAnswers[index] = new double[numOutputActivations];
      } // if (training)
      return;
   } //void allocateArrayMemory()

/**
 * IF RUNNING: Populates the weights with random values, unless the network is not in weightsConfiguration mode
 *    in which it manually loads the weights. All other arrays (inputs, hiddens, output) are auto set to 0.0.
 *
 * IF TRAINING: Populates the weights with random values, unless the network is not in weightsConfiguration mode
 *     in which it manually loads the weights. All other arrays (inputs, hiddens, output, thetas) are set to 0.0.
 */
   void populateArrays()
   {
      int I, J, K, M;

      if (weightsConfiguration == 0) // Randomizing Weights
      {
         for (I = 0; I < numOutputActivations; ++I)
            for (J = 0; J < numHiddenActivations[HIDDEN_LAYER_2]; ++J)
               weightsIJ[I][J] = randomValue();
         for (J = 0; J < numHiddenActivations[HIDDEN_LAYER_2]; ++J)
            for (K = 0; K < numHiddenActivations[HIDDEN_LAYER_1]; ++K)
               weightsJK[J][K] = randomValue();
         for (K = 0; K < numHiddenActivations[HIDDEN_LAYER_1]; ++K)
            for (M = 0; M < numInputActivations; ++M)
               weightsKM[K][M] = randomValue();
      } // if (weightsConfiguration == 0)
      else if (weightsConfiguration == 1) // Loading Weights
      {
         loadWeights();
      } // else if (weightsConfiguration == 1)
      else // Manually loading weights
      {
         if (numInputActivations == 2 && numHiddenActivations[HIDDEN_LAYER_1] == 5 && numHiddenActivations[HIDDEN_LAYER_2] == 3
            && numOutputActivations == 3)
         {
            weightsIJ[0][0] = 0.0; weightsIJ[0][1] = 0.0; weightsIJ[0][2] = 0.0; weightsIJ[0][3] = 0.0; weightsIJ[0][4] = 0.0;
            weightsIJ[1][0] = 0.0; weightsIJ[1][1] = 0.0; weightsIJ[1][2] = 0.0; weightsIJ[1][3] = 0.0; weightsIJ[1][4] = 0.0;
            weightsIJ[2][0] = 0.0; weightsIJ[2][1] = 0.0; weightsIJ[2][2] = 0.0; weightsIJ[2][3] = 0.0; weightsIJ[2][4] = 0.0;

            weightsJK[0][0] = 0.0; weightsJK[0][1] = 0.0; weightsJK[0][2] = 0.0; weightsJK[0][3] = 0.0; weightsJK[0][4] = 0.0;
            weightsJK[1][0] = 0.0; weightsJK[1][1] = 0.0; weightsJK[1][2] = 0.0; weightsJK[1][3] = 0.0; weightsJK[1][4] = 0.0;
            weightsJK[2][0] = 0.0; weightsJK[2][1] = 0.0; weightsJK[2][2] = 0.0; weightsJK[2][3] = 0.0; weightsJK[2][4] = 0.0;

            weightsKM[0][0] = 0.0; weightsKM[0][1] = 0.0;
            weightsKM[1][0] = 0.0; weightsKM[1][1] = 0.0;
            weightsKM[2][0] = 0.0; weightsKM[2][1] = 0.0;
            weightsKM[3][0] = 0.0; weightsKM[3][1] = 0.0;
         } // if (numInputActivations ...
         else
         {
            cout << "Manually set weights only available for 2-5-3 network. All weights set to default (0.0)" << endl;
         } // else (numInputActivations != 2...
      } // else (weightsConfiguration != 0 && weightsConfiguration != 1)

      if (training) // Populating Training Data and Training Answers
      {
         initializeTrainingData();
         initalizeTrainingAnswers();
      } // if (training)
      initializeTestingData(); // Populating Testing Data
      return;
   } //void populateArrays()

/**
 * Loads the weights from the fileWeights file
 */
   void loadWeights()
   {
      ifstream file;
      int I, J, K, M;
      file.open(fileWeights);
      string configuration = to_string(numInputActivations) + "_" + to_string(numHiddenActivations[HIDDEN_LAYER_1]) + "_" +
         to_string(numHiddenActivations[HIDDEN_LAYER_2]) + "_" + to_string(numOutputActivations);

      if (file.is_open())
      {
         string firstLine;
         getline(file, firstLine);
         if (firstLine == configuration)
         {
            string valueString;
            double value;

            for (M = 0; M < numHiddenActivations[HIDDEN_LAYER_1]; M++)
            {
               for (K = 0; K < numInputActivations; K++)
               {
                  getline(file, valueString);
                  value = stod(valueString);
                  weightsKM[M][K] = value;
               } // for (K = 0; K < numInputActivations; K++)
            } // for (M = 0; M < numHiddenActivations[HIDDEN_LAYER_1]; M++

            for (K = 0; K < numHiddenActivations[HIDDEN_LAYER_1]; K++)
            {
               for (J = 0; J < numHiddenActivations[HIDDEN_LAYER_2]; J++)
               {
                  getline(file, valueString);
                  value = stod(valueString);
                  weightsJK[J][K] = value;
               } // for (J = 0; J < numHiddenActivations[HIDDEN_LAYER_2]; J++)
            } // for (K = 0; K < numHiddenActivations[HIDDEN_LAYER_1]; K++)

            for (J = 0; J < numHiddenActivations[HIDDEN_LAYER_2]; J++)
            {
               for (I = 0; I < numOutputActivations; I++)
               {
                  getline(file, valueString);
                  value = stod(valueString);
                  weightsIJ[I][J] = value;
               } // for (I = 0; I < numOutputActivations; I++)
            } // for (J = 0; J < numHiddenActivations[HIDDEN_LAYER_2]; J++)
         } // if (firstLine == configuration)
         else // (firstLine != configuration)
         {
            cout << "File configuration does not match - all weights are set to default (0.0)" << endl;
         } // else (firstLine != configuration)
         file.close();
      } // if (file.is_open())

      else // (!file.is_open())
      {
         cout << "File not loaded - all weights are set to default (0.0)" << endl;
      } // else (!file.is_open())
      return;
   } // void loadWeights()

/**
 * Initializes the training answers based on the training answers file
 */
   void initializeTestingData()
   {
      ifstream file (testingFile);
      int caseNum, K;
      string configuration = to_string(numInputActivations);

      if (file.is_open())
      {
         string firstLine;
         getline(file, firstLine);
         if (firstLine == configuration)
         {
            string valueString;
            double value;

            for (caseNum = 0; caseNum < numCases; caseNum++)
            {
               for (K = 0; K < numInputActivations; K++)
               {
                  getline(file, valueString);
                  value = stod(valueString);
                  testData[caseNum][K] = value;
               } // for (K = 0; K < numInputActivations; K++)
            } // for (caseNum = 0; caseNum < numCases; caseNum++)
         } // if (firstLine == configuration)
         else // (firstLine != configuration)
         {
            cout << "File configuration does not match - all testing data set to default (0.0)" << endl;
         } // else (firstLine != configuration)
         file.close();
      } // if (file.is_open())
      else
      {
         cout << "Error - Cannot Open File for Testing Data" << endl;
      } // else (!file.is_open())
      return;
   } // void initializeTestingData()

/**
 * Initializes the training answers based on the training answers file
 */
   void initializeTrainingData()
   {
      ifstream file;
      int caseNum, K;
      file.open(trainingFile);
      string configuration = to_string(numInputActivations);

      if (file.is_open())
      {
         string firstLine;
         getline(file, firstLine);
         if (firstLine == configuration)
         {
            string valueString;
            double value;

            for (caseNum = 0; caseNum < numCases; caseNum++)
            {
               for (K = 0; K < numInputActivations; K++)
               {
                  getline(file, valueString);
                  value = stod(valueString);
                  trainData[caseNum][K] = value;
               } // for (K = 0; K < numInputActivations; K++)
            } // for (D = 0; D < numCases; D++)

         } // if (firstLine == configuration)
         else // (firstLine != configuration)
         {
            cout << "File configuration does not match - all training data set to default - " <<
                    "Abandoning Training" << endl;
            training = false;
         } // else (firstLine != configuration)
         file.close();
      } // if (file.is_open())
      else // (!file.is_open())
      {
         cout << "Error - Cannot Open File for Training Data" << endl;
      } // else (!file.is_open())
      return;
   } // void initializeTrainingData()

/**
 * Initializes the training answers based on the training answers file
 */
   void initalizeTrainingAnswers()
   {
      ifstream file;
      int caseNum, I;
      file.open(trainingAnswersFile);
      string configuration = to_string(numOutputActivations);

      if (file.is_open())
      {
         string firstLine;
         getline(file, firstLine);
         if (firstLine == configuration)
         {
            string valueString;
            double value;

            for (caseNum = 0; caseNum < numCases; caseNum++)
            {
               for (I = 0; I < numOutputActivations; I++)
               {
                  getline(file, valueString);
                  value = stod(valueString);
                  trainAnswers[caseNum][I] = value;
               } // for (K = 0; K < numInputActivations; K++)
            } // for (D = 0; D < numCases; D++)
         } // if (firstLine == configuration)
         else // (firstLine != configuration)
         {
            cout << "File configuration does not match - all training answers set to default - " <<
                    "Abandoning Training" << endl;
            training = false;
         } // else (firstLine != configuration)
         file.close();
      } // if (file.is_open())
      else
      {
         cout << "Error - Cannot Open File for Training Answers" << endl;
      } // else (!file.is_open())
      return;
   } // void initalizeTrainingAnswers()

/**
 * Outputs the Network Type, Lambda Value, Error Threshold, Maximum Number of Iterations,
 *    and the Random Number Range. To be used before training and/or running.
 */
   void checkNetwork()
   {
      if (training)
      {
         cout << "Training the Network:" << endl;
         cout << "   " << "Network Type: " << numInputActivations << "-" << numHiddenActivations[HIDDEN_LAYER_1] << "-" <<
            numHiddenActivations[HIDDEN_LAYER_2] << "-" << numOutputActivations << endl;
         cout << "   " << "Lambda Value: " << lambda << endl;
         cout << "   " << "Error Threshold: " << errorThreshold << endl;
         cout << "   " << "Maximum Number of Iterations: " << maxIterations << endl;
         cout << "   " << "Random Value Minimum: " << randMin << endl;
         cout << "   " << "Random Value Maximum: " << randMax << endl;
         cout << endl;
      } // if (training)
      else
      {
         cout << "Running with Network Type: " << numInputActivations << "-" << numHiddenActivations[HIDDEN_LAYER_1] << "-" <<
            numHiddenActivations[HIDDEN_LAYER_2] << "-" << numOutputActivations << endl;
      } // else (!training)
      return;
   } //void checkNetwork()

/**
 * Runs the network using predetermined test data. Used for solely running purposes.
 *     Each node is calculated using the activation function function applied onto a dot product
 *     of the weights and the activations.
 */
   double* run(double *inputValues)
   {
      time_t dummyStart, dummyEnd;
      int I, J, K, M;
      double thetaJ, thetaI, thetaK;

      time(&dummyStart);

      a = inputValues;

      for (K = 0; K < numHiddenActivations[HIDDEN_LAYER_1]; ++K)
      {
         thetaK = 0.0;
         for (M = 0; M < numInputActivations; ++M)
         {
            thetaK += a[M] * weightsKM[K][M];
         } // for (M = 0; M < numInputActivations; ++M)
         h[HIDDEN_LAYER_1][K] = activationFunction(thetaK);
      }

      for (J = 0; J < numHiddenActivations[HIDDEN_LAYER_2]; ++J)
      {
         thetaJ = 0.0;
         for (K = 0; K < numHiddenActivations[HIDDEN_LAYER_1]; ++K)
         {
            thetaJ += a[K] * weightsJK[J][K];
         } // for (K = 0; K < numHiddenActivations[HIDDEN_LAYER_1]; ++K)
         h[HIDDEN_LAYER_2][J] = activationFunction(thetaJ);
      } // for (J = 0; J < numHiddenActivations[HIDDEN_LAYER_2]; ++J)

      for (I = 0; I < numOutputActivations; ++I)
      {
         thetaI = 0.0;
         for (J = 0; J < numHiddenActivations[HIDDEN_LAYER_2]; ++J)
         {
            thetaI += h[HIDDEN_LAYER_2][J] * weightsIJ[I][J];
         } // for (J = 0; J < numHiddenActivations[HIDDEN_LAYER_2]; ++J)
         F[I] = activationFunction(thetaI);
      } // for (I = 0; I < numOutputActivations; ++I)

      time(&dummyEnd);
      runningTime = double(dummyEnd - dummyStart);
      return F;
   } // double run(double *inputValues)

/**
 * Runs the network ONLY DURING TRAINING using predetermined test data. Each node is calculated using
 *    the activation function function applied onto a dot product of the weights and the activations.
 */
   double* runTrain(double *inputValues, double *answersArray)
   {
      int I, J, K, M;
      double thetaI, lowerOmega;

      a = inputValues;

      for (K = 0; K < numHiddenActivations[HIDDEN_LAYER_1]; ++K)
      {
         thetaK[K] = 0.0;
         for (M = 0; M < numInputActivations; ++M)
         {
            thetaK[K] += a[M] * weightsKM[K][M];
         } // for (M = 0; M < numInputActivations; ++M)
         h[HIDDEN_LAYER_1][K] = activationFunction(thetaK[K]);
      }

      for (J = 0; J < numHiddenActivations[HIDDEN_LAYER_2]; ++J)
      {
         thetaJ[J] = 0.0;
         for (K = 0; K < numHiddenActivations[HIDDEN_LAYER_1]; ++K)
         {
            thetaJ[J] += a[K] * weightsJK[J][K];
         } // for (K = 0; K < numHiddenActivations[HIDDEN_LAYER_1]; ++K)
         h[HIDDEN_LAYER_2][J] = activationFunction(thetaJ[J]);
      } // for (J = 0; J < numHiddenActivations[HIDDEN_LAYER_2]; ++J)

      for (I = 0; I < numOutputActivations; ++I)
      {
         thetaI = 0.0;
         for (J = 0; J < numHiddenActivations[HIDDEN_LAYER_2]; ++J)
         {
            thetaI += h[HIDDEN_LAYER_2][J] * weightsIJ[I][J];
         } // for (J = 0; J < numHiddenActivations[HIDDEN_LAYER_2]; ++J)
         F[I] = activationFunction(thetaI);
         lowerOmega = answersArray[I] - F[I];
         lowerPsi[I] = lowerOmega * activationFunctionDerivative(thetaI);
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
      int I, J, K, M, caseNum;
      double capitalOmegaJ, capitalOmegaK, lowerOmega, *capitalPsiJ, *capitalPsiK, *testingArray, *answersArray;

      time(&dummyStart);
      capitalPsiJ = new double[numHiddenActivations[HIDDEN_LAYER_2]];
      capitalPsiK = new double[numHiddenActivations[HIDDEN_LAYER_1]];
      errorReached = DBL_MAX;

      checkNetwork();

      while (iterations < maxIterations && errorReached > errorThreshold)
      {
         errorReached = 0.0;
         for (caseNum = 0; caseNum < numCases; ++caseNum)
         {
            testingArray = trainData[caseNum];
            answersArray = trainAnswers[caseNum];

            runTrain(testingArray, answersArray);

            for (J = 0; J < numHiddenActivations[HIDDEN_LAYER_2]; ++J)
            {
               capitalOmegaJ = 0.0;
               for (I = 0; I < numOutputActivations; ++I)
               {
                  capitalOmegaJ += lowerPsi[I] * weightsIJ[I][J];
                  weightsIJ[I][J] += lambda * h[HIDDEN_LAYER_2][J] * lowerPsi[I];
               } // for (I = 0; I < numOutputActivations; ++I)
               capitalPsiJ[J] = capitalOmegaJ * activationFunctionDerivative(thetaJ[J]);
            } // for (J = 0; J < numHiddenActivations[HIDDEN_LAYER_2]; ++J)

            for (K = 0; K < numHiddenActivations[HIDDEN_LAYER_1]; ++K)
            {
               capitalOmegaK = 0.0;
               for (J = 0; J < numHiddenActivations[HIDDEN_LAYER_2]; ++J)
               {
                  capitalOmegaK += capitalPsiJ[J] * weightsJK[J][K];
                  weightsJK[J][K] += lambda * h[HIDDEN_LAYER_1][K] * capitalPsiJ[J];
               } // for (J = 0; J < numHiddenActivations[HIDDEN_LAYER_2]; ++J)

               capitalPsiK[K] = capitalOmegaK * activationFunctionDerivative(thetaK[K]);
               for (M = 0; M < numInputActivations; ++M)
               {
                  weightsKM[K][M] += lambda * a[M] * capitalPsiK[K];
               } // for (M = 0; M < numInputActivations; ++M)
            } // for (K < 0; K < numHiddenActivations[HIDDEN_LAYER_1]; ++K)

            run(testingArray);
            for (I = 0; I < numOutputActivations; ++I)
            {
               lowerOmega = answersArray[I] - F[I];
               errorReached += 0.5 * lowerOmega * lowerOmega;
            } // for (I = 0; I < numOutputActivations; ++I)
         } // for (caseNum = 0; caseNum < numCases; ++caseNum)

         errorReached /= ((double) numCases);
         ++iterations;
      } // while (epoch < maxIterations && errorReached > errorThreshold)
      time(&dummyEnd);
      trainingTime = int(dummyEnd) - int(dummyStart);
      return;
   } // void train()

/**
 * Saves the weights to a file, which is called "WeightsI" + n x "I", where n is the number of files already created.
 * It adds the configuration of the network to the first line of the file, and then adds the weights to the file.
 */
   void saveWeights()
   {
      ifstream file;
      string str;
      int I, J, K, M;
      string fileName = saveWeightsFile;

      file.open(fileName);

      while (file) // Making sure a new file is created
      {
         file.close();
         fileName += "I";
         file.open(fileName);
      } // while (file)

      ofstream outfile(fileName);

      outfile << to_string(numInputActivations) << "-" << to_string(numHiddenActivations[HIDDEN_LAYER_1]) << "-" <<
            to_string(numHiddenActivations[HIDDEN_LAYER_2]) << "-" << to_string(numOutputActivations) << endl;

      for (M = 0; M < numInputActivations; M++)
      {
         for (K = 0; K < numHiddenActivations[HIDDEN_LAYER_1]; K++)
         {
            outfile << fixed << setprecision(outputPrecision) << weightsKM[K][M] << endl;
         } // for (K = 0; K < numInputActivations; K++)
      } // for (M = 0; M < numInputActivations; M++)

      for (K = 0; K < numHiddenActivations[HIDDEN_LAYER_1]; K++)
      {
         for (J = 0; J < numHiddenActivations[HIDDEN_LAYER_2]; J++)
         {
            outfile << fixed << setprecision(outputPrecision) << weightsJK[J][K] << endl;
         } // for (J = 0; J < numHiddenActivations[HIDDEN_LAYER_2]; J++)
      } // for (K = 0; K < numHiddenActivations[HIDDEN_LAYER_1]; K++)

      for (J = 0; J < numHiddenActivations[HIDDEN_LAYER_2]; J++)
      {
         for (I = 0; I < numOutputActivations; I++)
         {
            outfile << fixed << setprecision(outputPrecision) << weightsIJ[I][J] << endl;
         } // for (I = 0; I < numOutputActivations; I++)
      } // for (J = 0; J < numHiddenActivations[HIDDEN_LAYER_2]; J++)

      outfile.close();
      return;
   } // void saveWeights()

/**
 * Prints array in a sequence for the purposes of reporting results after running. Follows the format
 *   value1, value2, value3, ... valueN
 */
   void printArray(double* arr, int length)
   {
      for (int index = 0; index < length; ++index)
      {
         cout << std::setprecision (outputPrecision) << arr[index];
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
         string reasonEndTraining;
         if (iterations == maxIterations)
         {
            reasonEndTraining = "Maximum Number of Iterations Reached";
         } // if (iterations == maxIterations)
         else
         {
            reasonEndTraining = "Error Threshold Reached";
         } // else (iterations != maxIterations)

         cout << "Reporting Results: " << endl;
         cout << "   " <<  "Reason for Termination: " << reasonEndTraining << endl;
         cout << "   " << "Training Time Taken: ";
         printTime(trainingTime);
         cout << "   " << "Error Reached: " << errorReached << endl;
         cout << "   " << "Iterations reached: " << iterations << endl << endl;

         cout << "Truth Table and Expected Outputs: " << endl;
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
 * Accepts input from the user and runs the network using the input values. SPECIFIC TO BOOLEAN ALGEBRA
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
 * Main function of the program - creates and configures the network, allocates and populates the arrays of the network,
 *       trains it, and then runs it for all test cases and reports the results.
 */
int main(int argc, char** argv)
{
   srand(time(nullptr));
   rand();

   if (argc > 1) configFilePath = argv[1];
   cout << configFilePath << endl;

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
      network.run(network.testData[0]);
      network.reportResults();
   } // if (network.training)

   if (network.savingWeights) network.saveWeights();

   if (!network.training) // Running the Network using test data
   {
      testingData(&network);
      network.reportResults();
   } // if (!network.training)
   return 0;
} // int main(int argc, char** argv)
