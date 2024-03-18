/*
 * Author: Varun Thvar
 * Date of Creation: 1 March 2023
 * Description: This is an implementation of a simple trainable A-B-C Network
 *              designed for OR, AND, and XOR, all at the same time. This network incorporates the
 *              backprogation algorithm to minimize the error by adjusting the weights by the
 *              derivative of the Error function.
 *
 * TOML: This file uses the TOML++ Library to parse the configuration file, which is a .toml file.
 *       The code to execute this via terminal is "g++ -I tomlplusplus/include -std=c++17 ABCBackPropagation.cpp", and
 *       the user has to have the tomlplusplus library in the same directory as the file.
 */

#include <cfloat>
#include <cmath>
#include <iostream>
#include <ctime>
#include <fstream>
#include <thread>
#include <toml++/toml.hpp> // TOML Implementation from Toml++ Library

#define stringify(name)          (# name)
#define MILLISECONDS_IN_SECOND   ((double) 1000.0)
#define SECONDS_IN_MINUTE        ((double) 60.0)
#define MINUTES_IN_HOUR          ((double) 60.0)
#define HOURS_IN_DAY             ((double) 24.0)
#define DAYS_IN_WEEK             ((double) 7.0)
#define CONFIG_FILE_PATH         "AB1/configAB1.toml"
#define TRUE_STRING              "true"

using namespace std;

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
 *   loadWeights() - Loads the weights from the fileWeights file
 *   initializeTestingData() - Initializes the testing data based on the testing file
 *   initializeTrainingData() - Initializes the training data based on the training file
 *   initalizeTrainingAnswers() - Initializes the training answers based on the training answers file
 *   saveWeights() - Saves the weights to a file, which is called "WeightsI" + n x "I", where n is the number of files
 *       already created. It adds the configuration of the network to the first line of the file, and then adds the
 *       weights to the file.
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
   /**
    * Configuration Parameters for the Network, such as the network configuration, number of cases, learning rate,
    *    error threshold, maximum number of iterations, random value range, and whether the network is in training mode.
    */
   int numOutputActivations;     // Number of Output Activations
   int numInputActivations;      // Number of Input Activations
   int numHiddenActivations;     // Number of Hidden Nodes in each Hidden Layer
   double lambda;                // Learning Rate - changes the effect the derivative of the error has on the weights
   double errorThreshold;        // Threshold for the error to reach during training
   int maxIterations;            // Maximum number of iterations during training
   double randMin;               // Minimum value of the random value assigned to weights
   double randMax;               // Maximum value of the random value assigned to weights
   bool training;                // Whether or not the network is in training mode (the alternative being running mode)

   /**
    * Arrays for the Network, including layers (a, h, F), weights, and training data,
    *    and backpropagation-relevant arrays such as thetaJ, and lowerPsi
    */
   double* a;                    // Array for Input Activations
   double* h;                    // Array for Hidden Activations
   double* F;                    // Output Value
   double** weightsIJ;           // Weights between the Input Layer and the Hidden Layers
   double** weightsJK;           // Weights between the Hidden Layers and the Output Layer
   double* thetaJ;               // Values used to calculating the hidden nodes - dot product of inputs and weights
   double* lowerPsi;             // Value of lowerOmega multiplied by the derivative of the sigmoid function
   double** trainData;           // Training Data (Inputs)
   double** trainAnswers;        // TrainingAnswers.txt (Expected Outputs)
   double** testData;            // Test Data (Inputs)

   /**
    * Relevant Information for Training and Running the Network, for the purposes of reporting results,
    *    or checking efficacy of the network
    */
   int numCases;                 // Number of Test Cases
   double trainingTime;          // Time taken for training the network
   double runningTime;           // Time taken for running the network
   int weightsConfiguration;    // 0 if the weights are randomized, 1 if loaded from file, 2 if manually set
   int iterations;               // Number of iterations taken during training
   double errorReached;          // Error value reached at the end of training or running
   bool savingWeights;           // True if the network  saves the weights at the end of training, false if otherwise
   string fileWeights;           // File path for loading the weights instead of randomizing
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
    * Sets the configuration parameters for the network
    */
   void setConfigurationParameters()
   {
      auto config = toml::parse_file(CONFIG_FILE_PATH); // Parsing the Configuration File

      numInputActivations = *(config.get(stringify(numInputActivations))-> value<int>());
      numHiddenActivations = *(config.get(stringify(numHiddenActivations))-> value<int>());
      numOutputActivations = *(config.get(stringify(numOutputActivations))-> value<int>());
      weightsConfiguration = *(config.get(stringify(weightsConfiguration))-> value<int>());

      training = *(config.get(stringify(training))-> value<bool>());
      numCases = *(config.get(stringify(numCases))-> value<int>());
      maxIterations = *(config.get(stringify(maxIterations))-> value<int>());
      lambda = *(config.get(stringify(lambda))-> value<double>());
      randMin = *(config.get(stringify(randMin))-> value<double>());
      randMax = *(config.get(stringify(randMax))-> value<double>());
      errorThreshold = *(config.get(stringify(errorThreshold))-> value<double>());
      savingWeights = *(config.get(stringify(savingWeights))-> value<bool>());

      fileWeights = *(config.get(stringify(fileWeights))-> value<string>());
      trainingFile = *(config.get(stringify(trainingFile))-> value<string>());
      trainingAnswersFile = *(config.get(stringify(trainingAnswersFile))-> value<string>());
      testingFile = *(config.get(stringify(testingFile))-> value<string>());
      return;
   } // void setConfigurationParameters()

   /**
    * Outputs the configuration parameters for the network to check if they are set correctly
    */
   void echoConfigurationParameters()
   {
      cout << endl << "Echoing Configuration Parameters:" << endl;
      cout << "   Network Type: " << numInputActivations << "-" << numHiddenActivations << "-" <<
         numOutputActivations << endl;
      cout << "   Input File: " << testingFile << endl;
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
      } // if (training)
      return;
   } //void echoConfigurationParameters()

   /**
    * IF RUNNING: Allocates memory for the activations, hidden nodes, and weights of the network
    * IF TRAINING: Allocates memory for the activations, hidden nodes, weights, delta weights,
    *     training data, and test data of the network
    */
   void allocateArrayMemory()
   {
      int I, J;

      a = new double[numInputActivations]; // Initializing input and hidden activations
      h = new double[numHiddenActivations];
      F = new double[numOutputActivations];

      weightsIJ = new double*[numOutputActivations]; // Initializing weights
      for (I = 0; I < numOutputActivations; ++I) weightsIJ[I] = new double[numHiddenActivations];
      weightsJK = new double*[numHiddenActivations];
      for (J = 0; J < numHiddenActivations; ++J) weightsJK[J] = new double[numInputActivations];

      testData = new double*[numCases]; // Initializing Test Data
      for (int index = 0; index < numCases; ++index) testData[index] = new double[numInputActivations];

      if (training) // Initializing Training Data and Training Answers and ThetaJ and LowerPsi
      {
         thetaJ = new double[numHiddenActivations];
         lowerPsi = new double[numOutputActivations];

         trainData = new double*[numCases]; // Initializing Training Data
         for (int index = 0; index < numCases; ++index) trainData[index] = new double[numInputActivations];
         trainAnswers = new double*[numCases];
         for (int index = 0; index < numCases; ++index) trainAnswers[index] = new double[numOutputActivations];
      } // if (training)

      cout << "Allocated Memory!" << endl;
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
      int I, J, K;

      if (weightsConfiguration == 0) // Randomizing Weights
      {
         for (I = 0; I < numOutputActivations; ++I)
            for (J = 0; J < numHiddenActivations; ++J)
               weightsIJ[I][J] = randomValue();
         for (J = 0; J < numHiddenActivations; ++J)
            for (K = 0; K < numInputActivations; ++K)
               weightsJK[J][K] = randomValue();
      } // if (weightsConfiguration == 0)
      else if (weightsConfiguration == 1) // Loading Weights
      {
         loadWeights();
      } // else if (weightsConfiguration == 1)
      else // Manually loading weights
      {
         if (numInputActivations == 2 && numHiddenActivations == 5 && numOutputActivations == 3)
         {
            weightsJK[0][0] = 0.1;
            weightsJK[0][1] = 0.2;
            weightsJK[1][0] = 0.3;
            weightsJK[1][1] = 0.4;
            weightsJK[2][0] = 0.5;
            weightsJK[2][1] = 0.6;
            weightsJK[3][0] = 0.7;
            weightsJK[3][1] = 0.8;
            weightsJK[4][0] = 0.9;
            weightsJK[4][1] = 1.0;

            weightsIJ[0][0] = 0.1;
            weightsIJ[0][1] = 0.4;
            weightsIJ[0][2] = 0.7;
            weightsIJ[0][3] = 1.0;
            weightsIJ[0][4] = 0.2;
            weightsIJ[1][0] = 0.3;
            weightsIJ[1][1] = 0.5;
            weightsIJ[1][2] = 0.6;
            weightsIJ[1][3] = 0.8;
            weightsIJ[1][4] = 0.9;
            weightsIJ[2][0] = 0.2;
            weightsIJ[2][1] = 0.3;
            weightsIJ[2][2] = 0.4;
            weightsIJ[2][3] = 0.5;
            weightsIJ[2][4] = 0.6;
         } // if (numInputActivations == 2 && numHiddenActivations == 2 && numOutputActivations == 1)
      } // else (weightsConfiguration != 0 && weightsConfiguration != 1)

      if (training) // Populating Training Data and Training Answers
      {
         initializeTrainingData();
         initalizeTrainingAnswers();
      } // if (training)
      initializeTestingData(); // Populating Testing Data

      cout << "Populated Arrays!" << endl;
      return;
   } //void populateArrays()

   /**
    * Loads the weights from the fileWeights file
    */
   void loadWeights()
   {
      ifstream file;
      int I, J, K;
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
            cout << "File configuration does not match - all weights are set to default (0.0)" << endl;
         } // else // (firstLine != configuration)
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
               } // for (J = 0; J < numHiddenActivations; J++)
            } // for (K = 0; K < numInputActivations; K++)
         } // if (firstLine == configuration)
         else // (firstLine != configuration)
         {
            cout << "File configuration does not match - all testing data set to default (0.0)" << endl;
         } // else // (firstLine != configuration)
         file.close();
      }
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
         } // else // (firstLine != configuration)
         file.close();
      }
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
      }
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
      if (!training)
         cout << "Running with Network Type: " << numInputActivations << "-" << numHiddenActivations << "-"
         << numOutputActivations << endl;
      if (training)
      {
         cout << "Training the Network:" << endl;
         cout << "   " << "Network Type: " << numInputActivations << "-" << numHiddenActivations << "-"
              << numOutputActivations << endl;
         cout << "   " << "Lambda Value: " << lambda << endl;
         cout << "   " << "Error Threshold: " << errorThreshold << endl;
         cout << "   " << "Maximum Number of Iterations: " << maxIterations << endl;
         cout << "   " << "Random Value Minimum: " << randMin << endl;
         cout << "   " << "Random Value Maximum: " << randMax << endl;
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
   double* runTrain(double *inputValues, double *answersArray)
   {
      int I, J, K;
      auto* thetaI = new double[numOutputActivations];
      auto* lowerOmega = new double[numOutputActivations];

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
         lowerOmega[I] = answersArray[I] - F[I];
         lowerPsi[I] = lowerOmega[I] * sigmoidPrime(thetaI[I]);
      } // for (I = 0; I < numOutputActivations; ++I)

      delete thetaI;
      delete lowerOmega;
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
      int I, J, K, caseNum;
      double *capitalOmega, *capitalPsi, *testingArray, *answersArray;

      time(&dummyStart);
      capitalOmega = new double[numHiddenActivations];
      capitalPsi = new double[numHiddenActivations];
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

            for (J = 0; J < numHiddenActivations; ++J)
            {
               capitalOmega[J] = 0.0;
               for (I = 0; I < numOutputActivations; ++I)
               {
                  capitalOmega[J] += lowerPsi[I] * weightsIJ[I][J];
                  weightsIJ[I][J] += lambda * h[J] * lowerPsi[I];
               } // for (I = 0; I < numOutputActivations; ++I)

               capitalPsi[J] = capitalOmega[J] * sigmoidPrime(thetaJ[J]);

               for (K = 0; K < numInputActivations; ++K)
               {
                  weightsJK[J][K] += lambda * a[K] * capitalPsi[J];
               } // for (K = 0; K < numInputActivations; ++K)
            } // for (J = 0; J < numHiddenActivations; ++J)

            run(testingArray);
            for (I = 0; I < numOutputActivations; ++I)
            {
               errorReached += 0.5 * (answersArray[I] - F[I]) * (answersArray[I] - F[I]);
            } // for (I = 0; I < numOutputActivations; ++I)
         } // for (D = 0; D ...

         errorReached /= ((double) numCases);
         ++iterations;
      } // while (epoch < maxIterations && errorReached > errorThreshold)
      time(&dummyEnd);
      trainingTime = double(dummyEnd) - double(dummyStart);

      return;
   } // void train()

   /**
    * Saves the weights to a file, which is called "WeightsI" + n x "I", where n is the number of files already created.
    * It adds the configuration of the network to the first line of the file, and then adds the weights to the file.
    */
   void saveWeights()
   {
      string fileName = fileWeights;
      ifstream file;
      int I, J, K;
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
      string reasonEndTraining;
      if (iterations == maxIterations)
      {
         reasonEndTraining = "Maximum Number of Iterations Reached";
      } // if (iterations == maxIterations)
      else
      {
         reasonEndTraining = "Error Threshold Reached";
      } // else (iterations != maxIterations)
      if (training)
      {
         cout << "Reporting Results:" << endl;
         cout << "   " <<  "Reason for Termination: " << reasonEndTraining << endl;
         cout << "   " << "Training Time Taken: ";
         printTime(trainingTime);
         cout << "   " << "Error Reached: " << errorReached << endl;
         cout << "   " << "Iterations reached: " << iterations << endl << endl;
         cout << "Truth Table and Expected Outputs:" << endl;

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
int main()
{
   srand(time(nullptr));
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

   if (network.savingWeights) network.saveWeights();

   network.training = false; // Running the Network using test data
   testingData(&network);
   network.reportResults();

   return 0;
} // int main(int argc, char *argv[])
