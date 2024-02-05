/*
 * Author: Varun Thvar
 * Date of Creation: 26 January 2023
 * Description: This is an implementation of a simple A-B-1 Network designed for simple boolean algebra problems.
 */

#include <cmath>
#include <iostream>
#include <time.h>

#define activationFunction(value)       sigmoid(value)
#define derivativeFunction(value)       sigmoidPrime(value)

#define numberActivations        (int) 2
#define numberHiddenLayers       (int) 1
#define numberHiddenNodes        (int) 2
#define MAX_ITERATIONS           (int) 100000
#define LAMBDA                   (double) 0.3
#define RANDOM_MIN               (double) -1.5
#define RANDOM_MAX               (double) 1.5
#define ERROR_THRESHOLD          (double) (2.0e-4)
#define MILLISECONDS_IN_SECOND   (double) 1000.0
#define SECONDS_IN_MINUTE        (double) 60.0
#define MINUTES_IN_HOUR          (double) 60.0
#define HOURS_IN_DAY             (double) 24.0
#define DAYS_IN_WEEK             (double) 7.0
#define TRAIN                    (bool) true
#define RANDOMIZE                (bool) true

using namespace std;

/**
 * Implements the linear function, which is defined by
 *    linear(x) = x
 */
double linear(double value)
{
   return value;
}

/**
 * Implements the derivative linear function, which is defined by
 *    linearPrime(x) = 1.0
 */
double linearPrime(double value)
{
   return 1.0;
}

/**
 * This implements the sigmoid function, which is defined by
 *    sigmoid(x) = 1/(1 + e^(-x))
 */
double sigmoid(double value)
{
   return 1/(1 + exp(-value));
}

/**
 * This implements the derivative of the sigmoid function, which is defined by
 *    (d/dx) sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)), where sigmoid is defined by
 *    sigmoid(x) = 1/(1 + e^(-x))
 */
double sigmoidPrime(double value)
{
    double sig = sigmoid(value);
    return sig * (1 - sig);
}

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
 *   Train() - Trains the network using predetermined training data. The training is done using the gradient
 *          descent algorithm, which is used to minimize the error by adjusting the weights derivative of the error
 *          with respect to the weights
 *   Run() - Runs the network using predetermined test data. Each node is calculated using the sigmoid function
 *          applied onto a dot product of the weights and the activations
 *   reportResults() - Reports the results of the training or running of the network, depending on the mode
 *          the network is in training mode or not
 *
 */
struct NeuralNetwork
{
   int numOutputActivations; // Number of Output Activations
   int numInputActivations; // Number of Input Activations
   int numHiddenActivations; // Number of Hidden Nodes in each Hidden Layer
   int numHiddenLayers; // Number of Hidden Layers
   double lambda; // Learning Rate - changes how much affect the derivative of the error has on the weights
   float errorThreshold; // Threshold for the error to reach during training
   int maxIterations; // Maximum number of iterations during training
   double randMin; // Minimum value of the random value assigned to weights
   double randMax; // Maximum value of the random value assigned to weights

   double* a; // Array for Input Activations
   double* h; // Array for Hidden Activations
   double F0; // Output Value
   double* weights0J; // Weights between the Input Layer and the Hidden Layers
   double** weightsJK; // Weights between the Hidden Layers and the Output Layer
   double** trainData; // Training Data (Inputs)
   double* trainAnswers; // Training Answers (Expected Outputs)
   double** testData; // Test Data (Inputs)
   int numCases; // Number of Test Cases

   bool training; // Whether or not the network is in training mode (the alternative being running mode)
   int I, J, K; // Variables used for looping through the outputs, hidden layer, and activations respectively
   int D; // Used to iterate over all the data points in a train case
   int epoch; // Used to iterate over all the test cases
   double* thetaJ; // Values used for calculating the hidden nodes - dot product of activations and  weights
   double* thetaI; // Values used for calculating the Output - dot product of hidden layers and corresponding weights

   float dummyError; // Dummy variable used for calculating the error
   double lowerOmega; // Value of T0 - F0
   double lowerPsi; // Value of lowerOmega * sigmoidPrime(thetaI)
   double EPrimeJ0; // Value of -1 * h[J] * lowerPsi -> indicates derivative of the error with respect to the weights0J
   double* deltaWj0; // Value of -1 * lambda * EPrimeJ0 -> indicates the change in weights0J
   double* capitalOmega; // Values of lowerPsi * weights0J[J]
   double* capitalPsi; // Values of capitalOmega * sigmoidPrime(thetaJ)
   double EPrimeJK; // Value of -1 * a[K] * capitalPsi -> indicates derivative of the error with respect to weightsJK
   double deltaWJK; // Value of -1 * lambda * EPrimeJK -> indicates the change in weightsJK

   double trainingTime; // Time taken for training the network
   double runningTime; // Time taken for running the network
   bool randomize; // Whether or not the network is in randomize mode
   time_t dummyStart; // Dummy variable used for the start of timing the training or running
   time_t dummyEnd; // Dummy variable used for the end of timing the training or running
   int iterations; // Number of iterations taken during training
   float error_reached; // Error value reached at the end of training or running
   string reasonForEndOfTraining; // Reason for ending training

   /**
    * Outputs a random value based on the range as given in the configuration parameters
    */
   double randomValue()
   {
      return ((double) rand() / (RAND_MAX)) * (randMax - randMin) + randMin;
   } // double randomValue()

   /**
    * Sets the configuration parameters for the network based on the following parameters:
    *   numAct: Number of Activations
    *   numHidLayer: Number of Hidden Layers
    *   numHidInEachLayer: Number of Hidden Nodes in each Hidden Layer
    *   lamb: Lambda Value
    *   errorThres: Error Threshold
    *   maxIter: Maximum number of iterations
    *   min: Minimum value of the random value assigned to weights
    *   max: Maximum value of the random value assigned to weights
    *   train: Whether or not the network is in training mode (the alternative being running mode)
    *   randomize: Whether or not the network is in randomize mode
    */
   void setConfigurationParameters(int numAct, int numHidLayer, int numHidInEachLayer, double lamb,
                                   float errorThres, int maxIter, double min, double max,
                                   bool train, bool randomize)
   {
      this->numOutputActivations = 1;
      this->numInputActivations = numAct;
      this->numHiddenLayers = numHidLayer;
      this->numHiddenActivations = numHidInEachLayer;

      this->lambda = lamb;
      this->errorThreshold = errorThres;
      this->maxIterations = maxIter;
      this->randMin = min;
      this->randMax = max;

      this->training = train;
      this->randomize = randomize;
      this->numCases = 4;
      return;
   } // void setConfigurationParameters(int numAct, int numHidLayer ...

   /**
   * Outputs the configuration parameters for the network
   */
   void echoConfigurationParameters()
   {
      cout << endl << "Echoing Configuration Parameters:" << endl;
      cout << "Network Type: " << numInputActivations << "-" << numHiddenActivations << "-" << 1 << endl;
      cout << "Lambda Value: " << lambda << endl;
      cout << "Error Threshold: " << errorThreshold << endl;
      cout << "Maximum Number of Iterations: " << maxIterations << endl;
      cout << "Random Value Minimum: " << randMin << endl;
      cout << "Random Value Maximum: " << randMax << endl;
      cout << "Randomizing Weights: " << (randomize ? "True" : "False") << endl;
      return;
   } //void echoConfigurationParameters()

   /**
    * Allocates memory for the activations, hidden nodes, and weights of the network
    */
   void allocateArrayMemory()
   {
      a = new double[numInputActivations];
      h = new double[numHiddenActivations];

      weights0J = new double[numHiddenActivations];
      weightsJK = new double*[numHiddenActivations];
      for (J = 0; J < numHiddenActivations; ++J) weightsJK[J] = new double[numInputActivations];

      capitalOmega = new double[numHiddenActivations];
      capitalPsi = new double[numHiddenActivations];

      thetaI = new double[numOutputActivations];
      thetaJ = new double[numHiddenActivations];

      deltaWj0 = new double[numHiddenActivations];

      trainData = new double*[numCases];
      for (int index = 0; index < numCases; ++index) trainData[index] = new double[numInputActivations];
      trainAnswers = new double[numCases];
      testData = new double*[numCases];
      for (int index = 0; index < numCases; ++index) testData[index] = new double[numInputActivations];

      cout << "Allocated Memory!" << endl;
      return;
   } //void allocateArrayMemory()

   /**
    * Populates the arrays with random values, unless randomize is set to false, in which it manually overrides
    *    the values to match a set of pre-determined values
    */
   void populateArrays()
   {
      if (randomize) // Randomizing Weights
      {
         for (J = 0; J < numHiddenActivations; ++J) weights0J[J] = randomValue();
         for (J = 0; J < numHiddenActivations; ++J) for (int K = 0; K < numInputActivations; ++K)
            weightsJK[J][K] = randomValue();
      }
      else // Manually Overriding Values
      {
         weights0J[0] = 1;
         weights0J[1] = 1;
         for (J = 0; J < numHiddenActivations; ++J) for (int K = 0; K < numInputActivations; ++K) weightsJK[J][K] = 1;
      }

      for (K = 0; K < numInputActivations; ++K) a[K] = 0; // Initializing Activations
      for (J = 0; J < numHiddenActivations; ++J) h[J] = 0; // Initializing Hidden Nodes
      F0 = 0; // Initializing Output

      for (J = 0; J < numHiddenActivations; ++J) thetaJ[J] = 0; // Initializing Thetas
      thetaI[0] = 0;

      for (J = 0; J < numHiddenActivations; ++J) capitalOmega[J] = 0; // Initializing capital Omega and Psi
      for (J = 0; J < numHiddenActivations; ++J) capitalPsi[J] = 0;

      trainData[0][0] = 0; // Initializing Training Data
      trainData[0][1] = 0;
      trainData[1][0] = 0;
      trainData[1][1] = 1;
      trainData[2][0] = 1;
      trainData[2][1] = 0;
      trainData[3][0] = 1;
      trainData[3][1] = 1;

      trainAnswers[0] = 0; // Initializing Training Answers
      trainAnswers[1] = 1;
      trainAnswers[2] = 1;
      trainAnswers[3] = 0;

      testData = trainData;
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
         cout << "Network Type: " << numInputActivations << "-" << numHiddenActivations << "-" << 1 << endl;
      if (training)
      {
         cout << "Training the Network!" << endl;
         cout << "Network Type: " << numInputActivations << "-" << numHiddenActivations << "-" << 1 << endl;
         cout << "Lambda Value: " << lambda << endl;
         cout << "Error Threshold: " << errorThreshold << endl;
         cout << "Maximum Number of Iterations: " << maxIterations << endl;
         cout << "Random Value Minimum: " << randMin << endl;
         cout << "Random Value Maximum: " << randMax << endl;
         cout << endl;
      }
      return;
   }

   /**
    * Trains the network using predetermined training data using the gradient descent algorithm, which is used to
    *    minimize the error by adjusting the weights derivative of the error with respect to the weights
    */
   void Train()
   {
      time(&dummyStart);
      checkNetwork();

      error_reached = INT_MAX;
      dummyError = 0;
      epoch = 0;

      while (epoch < maxIterations && error_reached > errorThreshold)
      {
         error_reached = 0;
         for (D = 0; D < 4; ++D)
         {
            F0 = Run(trainData[D]);

            dummyError = 0.5 * pow((trainAnswers[D] - F0), 2);

            lowerOmega = trainAnswers[D] - F0;
            lowerPsi = lowerOmega * derivativeFunction(thetaI[0]);
            for (J = 0; J < numHiddenActivations; ++J)
            {
               EPrimeJ0 = -1 * h[J] * lowerPsi;
               deltaWj0[J] = -1 * lambda * EPrimeJ0;
            } // for (J = 0; J < numHiddenActivations; ++J)

            for (J = 0; J < numHiddenActivations; ++J)
            {
               capitalOmega[J] = lowerPsi * weights0J[J];
               capitalPsi[J] = capitalOmega[J] * derivativeFunction(thetaJ[J]);

               for (K = 0; K < numInputActivations; ++K)
               {
                  EPrimeJK = -1 * a[K] * capitalPsi[J];
                  deltaWJK = -1 * lambda * EPrimeJK;
                  weightsJK[J][K] += deltaWJK;
               } // for (K = 0; K < numInputActivations; ++K)

               weights0J[J] += deltaWj0[J];
            } // for (J = 0; J < numHiddenActivations; ++J)

            dummyError = 0.5 * pow((trainAnswers[D] - F0), 2);
            error_reached += dummyError;
            F0 = Run(trainData[D]);
         } // for (D = 0; D ...

         error_reached /= numCases;
         ++epoch;
      } // while (epoch < maxIterations && error_reached > errorThreshold)

      if (epoch == maxIterations) reasonForEndOfTraining = "Maximum Number of Iterations Reached";
      else reasonForEndOfTraining = "Error Threshold Reached";

      trainingTime = double(time(&dummyEnd) - dummyStart);
      iterations = epoch;

      return;
   } // void Train()

   /**
    * Runs the network using predetermined test data. Each node is calculated using the sigmoid function applied
    *    onto a dot product of the weights and the activations
    */
   double Run(double *inputValues)
   {
      time(&dummyStart);
      a = inputValues;
      for (J = 0; J < numHiddenActivations; ++J)
      {
         thetaJ[J] = 0;
         for (K = 0; K < numInputActivations; ++K)
         {
            thetaJ[J] += a[K] * weightsJK[J][K];
         } // for (K = 0; K < numInputActivations; ++K)
         h[J] = activationFunction(thetaJ[J]);
      } // for (J = 0; J < numHiddenActivations; ++J)

      thetaI[0] = 0;
      for (J = 0; J < numHiddenActivations; ++J)
      {
         thetaI[0] += h[J] * weights0J[J];
      } // for (J = 0; J < numHiddenActivations; ++J)
      F0 = activationFunction(thetaI[0]);

      time(&dummyEnd);
      runningTime = double(dummyEnd - dummyStart);

      return F0;
   } // void Run(double inputValues[])

   /**
    * Reports the results of the training or running of the network, depending on the mode the network
    *    is in training mode or not
    */
   void reportResults()
   {
      if (training)
      {
         cout << "Reason for Termination: " << reasonForEndOfTraining << endl;
         cout << "Training Time Taken: ";
         printTime(trainingTime);
         cout << "Error Reached: " << error_reached << endl;
         cout << "Iterations reached: " << iterations << endl;
         cout << "Truth Table" << endl;
         for (int index = 0; index < numCases; ++index)
         {
            cout << trainData[index][0] << " & " << trainData[index][1] << " = " << trainAnswers[index] << " -> " << Run(trainData[index]) << endl;
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
      cout << "Running the Network with Test Data: " << network->testData[index][0] << " & " << network->testData[index][1] << endl;
      cout << network->Run(network->testData[index]) << endl << endl;
   } // for (int numOutputActivations = 0; numOutputActivations < 4; ++numOutputActivations)
   return;
} // void testingData(NeuralNetwork* network)

/**
 * Main function of the program - creates and configures the network, trains it, and then runs it for all test cases
 */
int main(int argc, char *argv[])
{
   srand(time(NULL));
   NeuralNetwork network; // Creating and Configurating the Network based on pre-determined constants and designs
   network.setConfigurationParameters(numberActivations, numberHiddenLayers,
      numberHiddenNodes, LAMBDA, ERROR_THRESHOLD, MAX_ITERATIONS,
      RANDOM_MIN, RANDOM_MAX, TRAIN, RANDOMIZE);
   network.echoConfigurationParameters();
   cout << endl;

   network.allocateArrayMemory(); // Allocating Arrays in Network
   network.populateArrays(); // Populating Arrays in Network
   cout << endl;

   if (network.training) // Training the Network using predetermined training data
   {
      network.Train();
      network.reportResults();
   }

   network.training = false; // Running the Network using test data
   testingData(&network);
   network.reportResults();

   return 0;
} // int main(int argc, char *argv[])
