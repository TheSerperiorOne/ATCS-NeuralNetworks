/*
 * Author: Varun Thvar
 * Date of Creation: 10 April 2024
 * Description: This is an implementation of a simple trainable N-Layer Network. This network incorporates the
 *              backpropagation algorithm to minimize the error by adjusting the weights by the
 *              derivative of the Error function.
 *
 * TOML: This file uses the TOML++ Library written by Mark Gillard to parse the configuration file, a .toml file.
 *       The code to execute this via terminal is "g++ -I tomlplusplus/include -std=c++20 NLayerNetwork.cpp", and
 *       the user has to have the tomlplusplus library in the same directory as the file.
 *
 * Table of Contents:
 *    1. double linear(double value)
 *    2. double linearPrime(double value)
 *    3. double sigmoid(double value)
 *    4. double sigmoidPrime(double value)
 *    5. void printTime(double seconds)
 *    6. struct NeuralNetwork
 *       6.1 double randomValue()
 *       6.2 void setConfigurationParameters()
 *       6.3 void printNetworkConfig()
 *       6.4 string networkConfig()
 *       6.5 void echoConfigurationParameters()
 *       6.6 void allocateArrayMemory()
 *       6.7 void populateArrays()
 *       6.8 void loadWeights()
 *       6.9 void initializeTestingData()
 *      6.10 void initializeTrainingData()
 *      6.11 void initializeTrainingAnswers()
 *      6.12 void checkNetwork()
 *      6.13 double* run(double *inputValues)
 *      6.14 double* runTrain(double *inputValues, double *answers)
 *      6.15 void train()
 *      6.16 void saveWeights()
 *    7. int main()
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
#define RANDOMIZE_WEIGHTS                  0
#define LOAD_WEIGHTS                       1


using namespace std;

/**
 * Implements the linear function, which is defined by
 *    linear(x) = x
 */
double linear(double value)
{
   return value;
} // double linear(double value)

/**
 * Implements the derivative of the linear function, which is defined by
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
   return 1.0 / (1.0 + exp(-value));
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
 * Implements the hyperbolic tangent function, which is defined by
 *    tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
 */
double tanh(double value)
{
   double sign = value > 0.0 ? -1.0 : 1.0;
   return sign * ((exp(sign * 2.0 * value) - 1.0) / (exp(sign * 2.0 * value) + 1.0));
} // double tanh(double value)

/**
 * Implements the derivative of the hyperbolic tangent function, which is defined by
 *    (d/dx) tanh(x) = 1 - tanh(x)^2, where tanh is defined by
 *    tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
 */
double tanhPrime(double value)
{
   double hyptan = tanh(value);
   return 1.0 - (hyptan * hyptan);
} // double tanhPrime(double value)

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
 * This is the structure of the Neural Network that is used to store the configuration parameters, the arrays
 *   for the network, and the relevant information for training and running the network.
 *   The structure also contains functions to set the configuration parameters, allocate memory for the arrays,
 *   populate the arrays, load the weights, initialize the training data, training answers, and testing data,
 *   run the network, run the network during training, train the network, and save the weights.
 *
 */
struct NeuralNetwork
{
/**
 * Configuration Parameters for the Network, such as the network configuration, number of cases, learning rate,
 *    error threshold, maximum number of iterations, random value range, and whether the network is in training mode.
 */
   int numLayers;                // Number of Layers
   int inputIndex;               // Index for LayerConfiguration input
   int outputIndex;              // Index for LayerConfiguration output
   int *LayerConfiguration;      // Configuration of the Layers
   double lambda;                // Learning Rate
   double errorThreshold;        // Threshold for the error to reach during training
   int maxIterations;            // Maximum number of iterations during training
   double randMin;               // Minimum value of the random value assigned to weights
   double randMax;               // Maximum value of the random value assigned to weights
   bool training;                // Whether the network is in training mode (true) or running mode (false)
   int outputPrecision;          // Precision of the output values

/**
 * Activation Function and its Derivative, which are used to calculate the hidden nodes and the output nodes
 */
   double (*activationFunction)(double value);             // Activation Function
   double (*activationFunctionDerivative)(double value);   // Derivative of the Activation Function

/**
 * Arrays for the Network, including layers, weights, and training data,
 *    and backpropagation-relevant arrays such as thetaJ, and lowerPsi
 */
   double **a;                   // Array for Layers Activations
   double ***weights;            // Weights between the Layers
   double **theta;               // Values used to calculating the hidden nodes - dot product of inputs and weights
   double **psi;                 // Value of used for training
   double **trainData;           // Training Data (Inputs)
   double **trainAnswers;        // TrainingAnswers.txt (Expected Outputs)
   double **testData;            // Test Data (Inputs)

/**
 * Relevant Information for Training and Running the Network, for the purposes of reporting results,
 *    or checking efficacy of the network
 */
   int numCases;                 // Number of Test Cases
   int trainingTime;             // Time taken for training the network
   double runningTime;           // Time taken for running the network
   int weightsConfiguration;     // 0 if the weights are randomized, if loaded from file, 2 if manually set
   int iterations;               // Number of iterations taken during training
   double errorReached;          // Error value reached at the end of training or running
   bool savingWeights;           // True if the network  saves the weights at the end of training, false if otherwise
   int keepAlive;                // Number of iterations for printing error
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
   void setConfigurationParameters(string configFilePath)
   {
      auto config = toml::parse_file(configFilePath);
      toml::array arr = *config.get_as<toml::array>("LayerConfiguration");
      
      numLayers = arr.size();
      inputIndex = 0;
      outputIndex = numLayers - 1;
      
      LayerConfiguration = new int[arr.size()];
      for (int index = 0; index < numLayers; ++index)
      {
         LayerConfiguration[index] = *arr[index].value<int>();
      } // for (int index = 0; index < numLayers; ++index)
      
      string activationFunctionString = *(config.get(stringify(activationFunction))->value<string>());
      if (activationFunctionString == stringify(linear))
      {
         activationFunction = linear;
         activationFunctionDerivative = linearPrime;
      } // if (activationFunctionString == stringify(linear))
      else if (activationFunctionString == stringify(sigmoid))
      {
         activationFunction = sigmoid;
         activationFunctionDerivative = sigmoidPrime;
      } // else if (activationFunctionString == stringify(sigmoid))
      else if (activationFunctionString == stringify(tanh))
      {
         activationFunction = tanh;
         activationFunctionDerivative = tanhPrime;
      } // else if (activationFunctionString == stringify(tanh))
      else
      {
         cout << "Activation Function not recognized - Defaulting to Sigmoid" << endl;
         activationFunction = sigmoid;
         activationFunctionDerivative = sigmoidPrime;
      } // else (activationFunctionString != stringify(linear) && activationFunctionString != stringify(sigmoid))
      
      weightsConfiguration = *(config.get(stringify(weightsConfiguration))->value<int>());
      outputPrecision = *(config.get(stringify(outputPrecision))->value<int>());
      training = *(config.get(stringify(training))->value<bool>());
      numCases = *(config.get(stringify(numCases))->value<int>());
      maxIterations = *(config.get(stringify(maxIterations))->value<int>());
      lambda = *(config.get(stringify(lambda))->value<double>());
      randMin = *(config.get(stringify(randMin))->value<double>());
      randMax = *(config.get(stringify(randMax))->value<double>());
      errorThreshold = *(config.get(stringify(errorThreshold))->value<double>());
      savingWeights = *(config.get(stringify(savingWeights))->value<bool>());
      keepAlive = *(config.get(stringify(keepAlive))->value<int>());
      
      fileWeights = *(config.get(stringify(fileWeights))->value<string>());
      saveWeightsFile = *(config.get(stringify(saveWeightsFile))->value<string>());
      trainingFile = *(config.get(stringify(trainingFile))->value<string>());
      trainingAnswersFile = *(config.get(stringify(trainingAnswersFile))->value<string>());
      testingFile = *(config.get(stringify(testingFile))->value<string>());
      return;
   } // void setConfigurationParameters()

/**
 * Outputs the configuration parameters for the network and returns it as a string
 */
   string networkConfig()
   {
      string networkConfig = "";
      for (int index = 0; index < numLayers; ++index)
      {
         networkConfig += to_string(LayerConfiguration[index]);
         if (index != outputIndex) networkConfig += "-";
      } // for (int index = 0; index < numLayers; ++index)
      return networkConfig;
   } // string networkConfig()

/**
 * Outputs the configuration parameters for the network to check if they are set correctly
 */
   void echoConfigurationParameters()
   {
      cout << endl << "Echoing Configuration Parameters:" << endl;
      cout << "   Number of Layers: " << numLayers << endl;
      cout << "   Network Configuration: " << networkConfig() << endl;
      cout << "   Input File: " << testingFile << endl;
      cout << "   Output Precision: " << outputPrecision << endl;
      cout << "   Activation Function: " << (activationFunction == linear ? "Linear" : "Sigmoid") << endl;
      if (!training) cout << "   Test Case File:" << testingFile << endl;
      if (weightsConfiguration == 1) cout << "   Loading Weights from: " << fileWeights << endl;
      if (training)
      {
         cout << "   Training Parameters:" << endl;
         cout << "     " << "Number of Cases: " << numCases << endl;
         cout << "     " << "Lambda Value: " << lambda << endl;
         cout << "     " << "Error Threshold: " << errorThreshold << endl;
         cout << "     " << "Maximum Number of Iterations: " << maxIterations << endl;
         cout << "     " << "Random Value Minimum: " << randMin << endl;
         cout << "     " << "Random Value Maximum: " << randMax << endl;
         cout << "     " << "Weights Generation: " << (weightsConfiguration == 0 ? "Randomize" :
                                                        weightsConfiguration == 1 ? "Load from File" : "Manually Set")
              << endl;
         cout << "     " << "Keep Alive Frequency: " << keepAlive << endl;
         cout << "     " << "Saving Weights: " << (savingWeights ? "True" : "False") << endl;
         if (savingWeights) cout << "     " << "Weights Saved to: " << fileWeights << endl;
         cout << "     " << "Training File: " << trainingFile << endl;
         cout << "     " << "Training Answers File: " << trainingAnswersFile << endl;
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
      int J, n;
      
      a = new double *[numLayers]; // Initializing input and hidden activations
      for (n = 0; n < numLayers; ++n) a[n] = new double[LayerConfiguration[n]];
      
      weights = new double **[numLayers]; // Initializing weights
      for (n = 1; n < numLayers; ++n)
      {
         weights[n] = new double *[LayerConfiguration[n]];
         for (J = 0; J < LayerConfiguration[n]; J++)
         {
            weights[n][J] = new double[LayerConfiguration[n - 1]];
         } // for (J = 0; J < LayerConfiguration[n]; J++)
      } // for (n = 1; n < numLayers; ++n)
      
      testData = new double *[numCases]; // Initializing Test Data
      for (int index = 0; index < numCases; ++index) testData[index] = new double[LayerConfiguration[inputIndex]];
      
      if (training) // Initializing Training Data and Training Answers and ThetaJ, ThetaK, and LowerPsi
      {
         theta = new double *[numLayers];
         for (n = 1; n < numLayers; ++n)
         {
            theta[n] = new double[LayerConfiguration[n]];
         } // for (n = 1; n < numLayers; ++n)
         
         psi = new double *[numLayers];
         for (n = 1; n < numLayers; ++n)
         {
            psi[n] = new double[LayerConfiguration[n]];
         } // for (n = 1; n < numLayers; ++n)
         
         trainData = new double *[numCases]; // Initializing Training Data
         for (int index = 0; index < numCases; ++index) trainData[index] = new double[LayerConfiguration[inputIndex]];
         
         trainAnswers = new double *[numCases];
         for (int index = 0; index < numCases; ++index) trainAnswers[index] = new double[LayerConfiguration[outputIndex]];
      } // if (training)
      return;
   } // void allocateArrayMemory()

/**
 * IF RUNNING: Populates the weights with random values, unless the network is not in weightsConfiguration mode
 *    in which it manually loads the weights. All other arrays are auto set to 0.0.
 *
 * IF TRAINING: Populates the weights with random values, unless the network is not in weightsConfiguration mode
 *     in which it manually loads the weights. All other arrays are set to 0.0.
 */
   void populateArrays()
   {
      int J, K, n;

      if (weightsConfiguration == RANDOMIZE_WEIGHTS) // Randomizing Weights
      {
         for (n = 1; n < numLayers; ++n)
         {
            for (J = 0; J < LayerConfiguration[n]; ++J)
            {
               for (K = 0; K < LayerConfiguration[n - 1]; ++K)
               {
                  weights[n][J][K] = randomValue();
               } // for (M = 0; M < LayerConfiguration[n - 1]; M++)
            } // for (K = 0; K < LayerConfiguration[n]; K++)
         } // for (n = 1; n < numLayers; ++n)
      } // if (weightsConfiguration == 0)
      
      else if (weightsConfiguration == LOAD_WEIGHTS) // Loading Weights
      {
         loadWeights();
      } // else if (weightsConfiguration == 1)
      else // Manually loading weights
      {
         if (networkConfig() == "2-5-5-3")
         {
            cout << "Manually Set Weights for 2-5-5-3 Network Not Configured" << endl;
         } // if (numInputActivations == 2...
         else
         {
            cout << "Manually set weights only available for 2-5-3 network. All weights set to default (0.0)" << endl;
         } // else (numInputActivations != 2...
      } // else (weightsConfiguration != 0 && weightsConfiguration != 1)
      
      if (training) // Populating Training Data and Training Answers
      {
         initializeTrainingData();
         initializeTrainingAnswers();
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
      int J, K, n;
      file.open(fileWeights);
      string configuration = networkConfig();
      
      if (file.is_open())
      {
         string firstLine;
         getline(file, firstLine);
         if (firstLine == configuration)
         {
            string valueString;
            double value;
            
            for (n = 1; n < numLayers; ++n)
            {
               for (J = 0; J < LayerConfiguration[n]; J++)
               {
                  for (K = 0; K < LayerConfiguration[n - 1]; K++)
                  {
                     getline(file, valueString);
                     value = stod(valueString);
                     weights[n][J][K] = value;
                  } // for (K = 0; K < LayerConfiguration[n - 1]; K++)
               } // for (J = 0; J < LayerConfiguration[n]; J++)
            } // for (n = 1; n < numLayers; ++n)
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
 * Initializes the testing data based on the testing data file
 */
   void initializeTestingData()
   {
      ifstream file(testingFile);
      int caseNum, M;
      string configuration = to_string(LayerConfiguration[inputIndex]);
      
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
               for (M = 0; M < LayerConfiguration[inputIndex]; M++)
               {
                  getline(file, valueString);
                  value = stod(valueString);
                  testData[caseNum][M] = value;
               } // for (M = 0; M < LayerConfiguration[inputIndex]; M++)
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
 * Initializes the training data based on the training data file
 */
   void initializeTrainingData()
   {
      ifstream file;
      int caseNum, M;
      file.open(trainingFile);
      string configuration = to_string(LayerConfiguration[inputIndex]);
      
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
               for (M = 0; M < LayerConfiguration[inputIndex]; M++)
               {
                  getline(file, valueString);
                  value = stod(valueString);
                  trainData[caseNum][M] = value;
               } // for (M = 0; M < LayerConfiguration[inputIndex]; M++)
            } // for (caseNum = 0; caseNum < numCases; caseNum++)
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
   void initializeTrainingAnswers()
   {
      ifstream file;
      int caseNum, I;
      file.open(trainingAnswersFile);
      string configuration = to_string(LayerConfiguration[outputIndex]);
      
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
               for (I = 0; I < LayerConfiguration[outputIndex]; I++)
               {
                  getline(file, valueString);
                  value = stod(valueString);
                  trainAnswers[caseNum][I] = value;
               } // for (I = 0; I < LayerConfiguration[outputIndex]; I++)
            } // for (caseNum = 0; caseNum < numCases; caseNum++)
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
   } // void initializeTrainingAnswers()

/**
 * Outputs the Network Type, Lambda Value, Error Threshold, Maximum Number of Iterations,
 *    and the Random Number Range. To be used before training and/or running.
 */
   void checkNetwork()
   {
      if (training)
      {
         cout << "Training the Network:" << endl;
         cout << "   " << "Network Type: " << networkConfig() << endl;
         cout << "   " << "Lambda Value: " << lambda << endl;
         cout << "   " << "Error Threshold: " << errorThreshold << endl;
         cout << "   " << "Maximum Number of Iterations: " << maxIterations << endl;
         cout << "   " << "Random Value Minimum: " << randMin << endl;
         cout << "   " << "Random Value Maximum: " << randMax << endl;
         cout << endl;
      } // if (training)
      else
      {
         cout << "Running with Network Type: " << networkConfig() << endl;
      } // else (!training)
      return;
   } //void checkNetwork()

/**
 * Runs the network using predetermined test data. Used for solely running purposes.
 *     Each node is calculated using the activation function function applied onto a dot product
 *     of the weights and the activations.
 */
   double *run(double *inputValues)
   {
      time_t dummyStart, dummyEnd;
      int J, K, n;
      double theta;
      time(&dummyStart);
      
      a[inputIndex] = inputValues;
      
      for (n = 1; n < numLayers; ++n)
      {
         for (J = 0; J < LayerConfiguration[n]; ++J)
         {
            theta = 0.0;
            for (K = 0; K < LayerConfiguration[n - 1]; ++K)
            {
               theta += a[n - 1][K] * weights[n][J][K];
            } // for (K = 0; K < LayerConfiguration[n - 1]; ++K)
            a[n][J] = activationFunction(theta);
         } // for (J = 0; J < LayerConfiguration[n]; ++J)
      } // for (n = 1; n < numLayers; ++n)
      
      time(&dummyEnd);
      runningTime = double(dummyEnd - dummyStart);
      return a[outputIndex];
   } // double run(double *inputValues)

/**
 * Runs the network ONLY DURING TRAINING using predetermined test data. Each node is calculated using
 *    the activation function function applied onto a dot product of the weights and the activations.
 */
   double *runTrain(double *inputValues, double *answers)
   {
      int K, J, n;
      
      a[inputIndex] = inputValues;
      
      for (n = 1; n < numLayers; ++n)
      {
         for (J = 0; J < LayerConfiguration[n]; ++J)
         {
            theta[n][J] = 0.0;
            for (K = 0; K < LayerConfiguration[n - 1]; ++K)
            {
               theta[n][J] += a[n - 1][K] * weights[n][J][K];
            } // for (K = 0; K < LayerConfiguration[n - 1]; ++K)
            a[n][J] = activationFunction(theta[n][J]);
         } // for (J = 0; J < LayerConfiguration[n]; ++J)
      } // for (n = 1; n < numLayers; ++n)
      
      n = outputIndex;
      for (J = 0; J < LayerConfiguration[n]; ++J)
      {
         theta[n][J] = 0.0;
         for (K = 0; K < LayerConfiguration[n - 1]; ++K)
         {
            theta[n][J] += a[n - 1][K] * weights[n][J][K];
         } // for (K = 0; K < LayerConfiguration[n - 1]; ++K)
         a[n][J] = activationFunction(theta[n][J]);
         psi[n][J] = (answers[J] - a[n][J]) * activationFunctionDerivative(theta[n][J]);
      } // for (J = 0; J < LayerConfiguration[n]; ++J)
      
      return a[outputIndex];
   } // double runTrain(double *inputValues, double *answers)

/**
 * Trains the network using predetermined training data using the gradient descent algorithm, which is used to
 *    minimize the error by adjusting the weights derivative of the error with respect to the weights. Uses
 *    runTrain() to calculate the activations and outputs of the network.
 */
   void train()
   {
      time_t dummyStart, dummyEnd;
      int I, J, K, M, X, caseNum, n;
      double omega, *testingArray, *answersArray;
      
      time(&dummyStart);
      errorReached = DBL_MAX;
      
      while (iterations < maxIterations && errorReached > errorThreshold)
      {
         errorReached = 0.0;
         for (caseNum = 0; caseNum < numCases; ++caseNum)
         {
            testingArray = trainData[caseNum];
            answersArray = trainAnswers[caseNum];
            
            runTrain(testingArray, answersArray);
            
            for (n = outputIndex; n > inputIndex + 2; --n)
            {
               for (K = 0; K < LayerConfiguration[n - 1]; ++K)
               {
                  omega = 0.0;
                  for (J = 0; J < LayerConfiguration[n]; ++J)
                  {
                     omega += psi[n][J] * weights[n][J][K];
                     weights[n][J][K] += lambda * a[n - 1][K] * psi[n][J];
                  } // for (J = 0; J < LayerConfiguration[n]; ++J)
                  psi[n - 1][K] = omega * activationFunctionDerivative(theta[n - 1][K]);
               } // for (K = 0; K < LayerConfiguration[n - 1]; ++K)
            } // for (n = outputIndex; n > inputIndex + 2; --n)
            
            n = inputIndex + 2;
            
            for (M = 0; M < LayerConfiguration[n - 1]; ++M)
            {
               omega = 0.0;
               for (K = 0; K < LayerConfiguration[n]; ++K)
               {
                  omega += psi[n][K] * weights[n][K][M];
                  weights[n][K][M] += lambda * a[n - 1][M] * psi[n][K];
               } // for (K = 0; K < LayerConfiguration[n]; ++K)
               
               psi[n - 1][M] = omega * activationFunctionDerivative(theta[n - 1][M]);
               
               for (X = 0; X < LayerConfiguration[inputIndex]; ++X)
               {
                  weights[n - 1][M][X] += lambda * a[inputIndex][X] * psi[n - 1][M]; // inputIndex = n - 2
               } // for (X = 0; X < LayerConfiguration[inputIndex]; ++X)
            } // for (M = 0; M < LayerConfiguration[n - 1]; ++M)
            
            run(testingArray);
            for (I = 0; I < LayerConfiguration[outputIndex]; ++I)
            {
               omega = answersArray[I] - a[outputIndex][I];
               errorReached += 0.5 * omega * omega;
            } // for (I = 0; I < LayerConfiguration[outputIndex]; ++I)
         } // for (caseNum = 0; caseNum < numCases; ++caseNum)
         errorReached /= ((double) numCases);
         ++iterations;
         if (keepAlive && iterations % keepAlive == 0) // Printing the error every keepAlive iterations
         {
            cout << "Iteration: " << iterations << " - Error: " << errorReached << " - Time: ";
            printTime(int(time(&dummyEnd)) - int(dummyStart));
         } // if (keepAlive && iterations % keepAlive == 0)
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
      int J, K, n;
      string fileName = saveWeightsFile;
      
      file.open(fileName);
      
      while (file) // Making sure a new file is created
      {
         file.close();
         fileName += "I";
         file.open(fileName);
      } // while (file)
      
      ofstream outfile(fileName);
      
      outfile << networkConfig() << endl;
      
      for (n = 1; n < numLayers; ++n)
      {
         for (K = 0; K < LayerConfiguration[n]; ++K)
         {
            for (J = 0; J < LayerConfiguration[n - 1]; ++J)
            {
               outfile << fixed << setprecision(outputPrecision) << weights[n][K][J] << endl;
            } // for (J = 0; J < LayerConfiguration[n - 1]; ++J)
         } // for (K = 0; K < LayerConfiguration[n]; ++K)
      } // for (n = 1; n < numLayers; ++n)
      
      outfile.close();
      return;
   } // void saveWeights()

/**
 * Prints array in a sequence for the purposes of reporting results after running. Follows the format
 *   value1, value2, value3, ... valueN. Specifically for arrays of doubles
 */
   void printArray(double *arr, int length)
   {
      for (int index = 0; index < length; ++index)
      {
         cout << std::setprecision(outputPrecision) << arr[index];
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
         cout << "   " << "Reason for Termination: " << reasonEndTraining << endl;
         cout << "   " << "Training Time Taken: ";
         printTime(trainingTime);
         cout << "   " << "Error Reached: " << errorReached << endl;
         cout << "   " << "Iterations reached: " << iterations << endl << endl;
         
         cout << "Truth Table and Expected Outputs: " << endl;
         
         for (int index = 0; index < numCases; ++index)
         {
            printArray(runTrain(trainData[index], trainAnswers[index]), LayerConfiguration[outputIndex]);
            cout << endl;
         } // for (int index = 0...
         cout << endl;
      } // if (training)
      return;
   } // reportResults()
}; // struct NeuralNetwork

/**
 * Runs the network using the given test data, and prints the output of the network. Used for testing purposes,
 *          and should only be used if the inputs can be put into .
 */
void runningAllTestingData(NeuralNetwork *network)
{
   int n;
   for (int index = 0; index < network->numCases; ++index)
   {
      network->printArray(network->run(network->testData[index]), network->LayerConfiguration[network->outputIndex]);
      cout << endl << endl;
   } // for (int index = 0; index < network->numCases; ++index)
   return;
} // void runningAllTestingData(NeuralNetwork* network)

/**
 * Main function of the program - creates and configures the network, allocates and populates the arrays of the network,
 *       trains it, and then runs it for all test cases and reports the results.
 */
int main(int argc, char **argv)
{
   srand(time(nullptr));
   rand();
   
   string configFilePath;
   
   if (argc > 1)
   {
      configFilePath = argv[1];
   } // if (argc > 1)
   else
   {
      configFilePath = DEFAULT_CONFIG_FILE;
   } // else (argc <= 1)
   
   NeuralNetwork network; // Creating and configuring the network based on pre-determined constants and designs
   network.setConfigurationParameters(configFilePath);
   network.echoConfigurationParameters();
   cout << endl;
   
   network.allocateArrayMemory(); // Allocating Arrays in Network
   network.populateArrays(); // Populating Arrays in Network

   if (network.training) // Training the Network using predetermined training data
   {
      network.checkNetwork();
      network.train();
      network.run(network.testData[0]);
      network.reportResults();
   } // if (network.training)

   if (network.savingWeights)
   {
      network.saveWeights();
   } // if (network.savingWeights)

   if (!network.training) // Running the Network using test data
   {
      runningAllTestingData(&network);
   } // if (!network.training)
   return 0;
} // int main(int argc, char** argv)
