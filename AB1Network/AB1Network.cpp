/*
 * Author: Varun Thvar
 * Date of Creation: 26 January 2023
 * Description: This is an implementation of a simple trainable A-B-1 Network
 *              designed for simple boolean algebra problems. This network incorporates the gradient descent algorithm
 *              to minimize the error by adjusting the weights by the derivative of the Error function.
 */

#include <cmath>
#include <iostream>
#include <time.h>

#define activationFunction(value)       sigmoid(value)
#define derivativeFunction(value)       sigmoidPrime(value)

#define NUMBER_ACTIVATIONS       ((int) 2)
#define NUMBER_HIDDEN_LAYERS     ((int) 1)
#define NUMBER_HIDDEN_NODES      ((int) 2)
#define MAX_ITERATIONS           ((int) 100000)
#define LAMBDA                   ((double) 0.3)
#define RANDOM_MIN               ((double) -1.5)
#define RANDOM_MAX               ((double) 1.5)
#define ERROR_THRESHOLD          ((double) (2.0e-4))
#define MILLISECONDS_IN_SECOND   ((double) 1000.0)
#define SECONDS_IN_MINUTE        ((double) 60.0)
#define MINUTES_IN_HOUR          ((double) 60.0)
#define HOURS_IN_DAY             ((double) 24.0)
#define DAYS_IN_WEEK             ((double) 7.0)
#define TRAIN                    ((bool) true)
#define RANDOMIZE                ((bool) true)

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
 * Implements the derivative linear function, which is defined by
 *    linearPrime(x) = 1.0
 */
double linearPrime(double value)
{
   return 1.0;
} // double linearPrime(double value)

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
   double lambda;                // Learning Rate - changes how much affect the derivative of the error has on the weights
   double errorThreshold;        // Threshold for the error to reach during training
   int maxIterations;            // Maximum number of iterations during training
   double randMin;               // Minimum value of the random value assigned to weights
   double randMax;               // Maximum value of the random value assigned to weights
   bool training;                // Whether or not the network is in training mode (the alternative being running mode)

   double* a;                    // Array for Input Activations
   double* h;                    // Array for Hidden Activations
   double F0;                    // Output Value
   double* weights0J;            // Weights between the Input Layer and the Hidden Layers
   double** weightsJK;           // Weights between the Hidden Layers and the Output Layer
   double** trainData;           // Training Data (Inputs)
   double* trainAnswers;         // Training Answers (Expected Outputs)
   double* thetaJ;               // Values used for calculating the hidden nodes - dot product of activations and  weights
   double* thetaI;               // Values used for calculating the Output - dot product of hidden layers and corresponding weights
   double** testData;            // Test Data (Inputs)
   double* deltaWj0;             // Value of -1 * lambda * EPrimeJ0 -> indicates the change in weights0J
   double* capitalOmega;         // Values of lowerPsi * weights0J[J]
   double* capitalPsi;           // Values of capitalOmega * sigmoidPrime(thetaJ)

   int numCases;                 // Number of Test Cases
   double trainingTime;          // Time taken for training the network
   double runningTime;           // Time taken for running the network
   bool randomize;               // Whether or not the network is in randomize mode
   int iterations;               // Number of iterations taken during training
   double error_reached;         // Error value reached at the end of training or running
   string reasonEndTraining;     // Reason for ending training

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
      numOutputActivations = 1;
      numCases = 4;

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

      return;
   } // void setConfigurationParameters(int numAct, int numHidLayer ...

   /**
   * Outputs the configuration parameters for the network to check if they are set correctly
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
    * IF RUNNING: Allocates memory for the activations, hidden nodes, and weights of the network
    * IF TRAINING: Allocates memory for the activations, hidden nodes, weights, delta weights,
    *              training data, and test data of the network
    */
   void allocateArrayMemory()
   {
      int I, J, K, D;

      if (training)
      {
         a = new double[numInputActivations]; // Initializing input and hidden activations
         h = new double[numHiddenActivations];

         weights0J = new double[numHiddenActivations]; // Initializing weights
         weightsJK = new double*[numHiddenActivations];
         for (J = 0; J < numHiddenActivations; ++J) weightsJK[J] = new double[numInputActivations];

         capitalOmega = new double[numHiddenActivations]; // Initializing capital Omega and Psi
         capitalPsi = new double[numHiddenActivations];

         thetaI = new double[numOutputActivations]; // Initializing Thetas
         thetaJ = new double[numHiddenActivations];

         deltaWj0 = new double[numHiddenActivations]; // Initializing delta weights

         trainData = new double*[numCases]; // Initializing Training Data
         for (int index = 0; index < numCases; ++index) trainData[index] = new double[numInputActivations];
         trainAnswers = new double[numCases];

         testData = new double*[numCases]; // Initializing Test Data
         for (int index = 0; index < numCases; ++index) testData[index] = new double[numInputActivations];

      } // if (training)

      if (!training)
      {
         a = new double[numInputActivations]; // Initializing input and hidden activations
         h = new double[numHiddenActivations];

         weights0J = new double[numHiddenActivations]; // Initializing weights
         weightsJK = new double*[numHiddenActivations];
         for (J = 0; J < numHiddenActivations; ++J) weightsJK[J] = new double[numInputActivations];

         testData = new double*[numCases]; // Initializing test data
         for (D = 0; D < numCases; ++D) testData[D] = new double[numInputActivations];

      } // if (!training)
      cout << "Allocated Memory!" << endl;
      return;
   } //void allocateArrayMemory()

   /**
    * IF RUNNING: Populates the weights with random values, unless the network is a 2-2-1 network, in which
    *             it manually overrides the values. All other arrays (inputs, hiddens, output) are auto set to 0.0.
    *
    * IF TRAINING: Populates the weights with random values, unless the network is a 2-2-1 network, in which it
    *              manually overrides the values. All other arrays (inputs, hiddens, output) thetas) are auto set to
    *              0.0.
    */
   void populateArrays()
   {
      int I, J, K;

      if (training)
      {
         if (randomize) // Randomizing Weights
         {
            for (J = 0; J < numHiddenActivations; ++J) weights0J[J] = randomValue();
            for (J = 0; J < numHiddenActivations; ++J) for (int K = 0; K < numInputActivations; ++K)
               weightsJK[J][K] = randomValue();
         } // if (randomize)
         else // Manually Overriding Values
         {
            weights0J[0] = 0.103;
            weights0J[1] = 0.23;
            for (J = 0; J < numHiddenActivations; ++J) for (int K = 0; K < numInputActivations; ++K) weightsJK[J][K] = 1.0;
         } // else

         for (K = 0; K < numInputActivations; ++K) a[K] = 0.0; // Initializing Activations
         for (J = 0; J < numHiddenActivations; ++J) h[J] = 0.0; // Initializing Hidden Nodes
         F0 = 0.0; // Initializing Output

         for (J = 0; J < numHiddenActivations; ++J) thetaJ[J] = 0.0; // Initializing Thetas
         thetaI[0] = 0.0;

         for (J = 0; J < numHiddenActivations; ++J) capitalOmega[J] = 0.0; // Initializing capital Omega and Psi
         for (J = 0; J < numHiddenActivations; ++J) capitalPsi[J] = 0.0;

         trainData[0][0] = 0.0; // Initializing Training Data
         trainData[0][1] = 0.0;
         trainData[1][0] = 0.0;
         trainData[1][1] = 1.0;
         trainData[2][0] = 1.0;
         trainData[2][1] = 0.0;
         trainData[3][0] = 1.0;
         trainData[3][1] = 1.0;

         trainAnswers[0] = 0.0; // Initializing Training Answers
         trainAnswers[1] = 1.0;
         trainAnswers[2] = 1.0;
         trainAnswers[3] = 1.0;

         testData = trainData;
      } // if (training)

      if (!training)
      {
         if (randomize) // Randomizing Weights
         {
            for (J = 0; J < numHiddenActivations; ++J) weights0J[J] = randomValue();
            for (J = 0; J < numHiddenActivations; ++J) for (int K = 0; K < numInputActivations; ++K)
               weightsJK[J][K] = randomValue();
         } // if (randomize)

         else // Manually Overriding Values
         {
            weights0J[0] = 0.103;
            weights0J[1] = 0.23;
            for (J = 0; J < numHiddenActivations; ++J) for (int K = 0; K < numInputActivations; ++K) weightsJK[J][K] = 1.0;
         } // else

         for (K = 0; K < numInputActivations; ++K) a[K] = 0.0; // Initializing Activations
         for (J = 0; J < numHiddenActivations; ++J) h[J] = 0.0; // Initializing Hidden Nodes
         F0 = 0.0; // Initializing Output

         testData[0][0] = 0.0; // Initializing Training Data
         testData[0][1] = 0.0;
         testData[1][0] = 0.0;
         testData[1][1] = 1.0;
         testData[2][0] = 1.0;
         testData[2][1] = 0.0;
         testData[3][0] = 1.0;
         testData[3][1] = 1.0;
      } // if (!training)

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
      } // if (training)
      return;
   } //void checkNetwork()

   /**
   * Runs the network using predetermined test data. Used for solely running purposes.
   *     Each node is calculated using the sigmoid function applied onto a dot product
   *     of the weights and the activations.
    */
   double run(double *inputValues)
   {
      time_t dummyStart, dummyEnd;
      time(&dummyStart);

      int I, J, K;
      double thetaJ;
      double thetaI;

      a = inputValues;

      for (J = 0; J < numHiddenActivations; ++J)
      {
         thetaJ = 0.0;
         for (K = 0; K < numInputActivations; ++K)
         {
            thetaJ += a[K] * weightsJK[J][K];
         } // for (K = 0; K < numInputActivations; ++K)
         h[J] = activationFunction(thetaJ);
      } // for (J = 0; J < numHiddenActivations; ++J)

      thetaI = 0.0;
      for (J = 0; J < numHiddenActivations; ++J)
      {
         thetaI += h[J] * weights0J[J];
      } // for (J = 0; J < numHiddenActivations; ++J)

      F0 = activationFunction(thetaI);

      time(&dummyEnd);
      runningTime = double(dummyEnd - dummyStart);
      return F0;
   } // double run(double *inputValues)

   /**
    * Runs the network ONLY DURING TRAINING using predetermined test data. Each node is calculated using
    *    the sigmoid function applied onto a dot product of the weights and the activations.
    */
   double runTrain(double *inputValues)
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
         h[J] = activationFunction(thetaJ[J]);
      } // for (J = 0; J < numHiddenActivations; ++J)

      thetaI[0] = 0.0;
      for (J = 0; J < numHiddenActivations; ++J)
      {
         thetaI[0] += h[J] * weights0J[J];
      } // for (J = 0; J < numHiddenActivations; ++J)

      F0 = activationFunction(thetaI[0]);
      return F0;
   } // double runTrain(double *inputValues)

   /**
    * Trains the network using predetermined training data using the gradient descent algorithm, which is used to
    *    minimize the error by adjusting the weights derivative of the error with respect to the weights. Uses
    *    runTrain() to calculate the activations and outputs of the network.
    */
   void train()
   {
      time_t dummyStart, dummyEnd;
      time(&dummyStart);
      checkNetwork();

      int I, J, K, D, epoch;
      double dummyError, lowerOmega, lowerPsi, EPrimeJ0, EPrimeJK, deltaWJK;

      error_reached = INT_MAX;
      dummyError = 0.0;
      epoch = 0;

      while (epoch < maxIterations && error_reached > errorThreshold)
      {
         error_reached = 0.0;
         for (D = 0; D < 4; ++D)
         {
            runTrain(trainData[D]);

            dummyError = 0.5 * (trainAnswers[D] - F0) * (trainAnswers[D] - F0);

            lowerOmega = trainAnswers[D] - F0;
            lowerPsi = lowerOmega * derivativeFunction(thetaI[0]);

            for (J = 0; J < numHiddenActivations; ++J)
            {
               EPrimeJ0 = -1.0 * h[J] * lowerPsi;
               deltaWj0[J] = -1.0 * lambda * EPrimeJ0;
            } // for (J = 0; J < numHiddenActivations; ++J)

            for (J = 0; J < numHiddenActivations; ++J)
            {
               capitalOmega[J] = lowerPsi * weights0J[J];
               capitalPsi[J] = capitalOmega[J] * derivativeFunction(thetaJ[J]);

               for (K = 0; K < numInputActivations; ++K)
               {
                  EPrimeJK = -1.0 * a[K] * capitalPsi[J];
                  deltaWJK = -1.0 * lambda * EPrimeJK;
                  weightsJK[J][K] += deltaWJK;
               } // for (K = 0; K < numInputActivations; ++K)

               weights0J[J] += deltaWj0[J];
            } // for (J = 0; J < numHiddenActivations; ++J)

            runTrain(trainData[D]);
            dummyError = 0.5 * (trainAnswers[D] - F0) * (trainAnswers[D] - F0);
            error_reached += dummyError;
         } // for (D = 0; D ...

         error_reached /= numCases;
         ++epoch;
      } // while (epoch < maxIterations && error_reached > errorThreshold)

      if (epoch == maxIterations) reasonEndTraining = "Maximum Number of Iterations Reached";
      else reasonEndTraining = "Error Threshold Reached";

      trainingTime = double(time(&dummyEnd) - dummyStart);
      iterations = epoch;

      return;
   } // void train()

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
         cout << "Error Reached: " << error_reached << endl;
         cout << "Iterations reached: " << iterations << endl << endl;
         cout << "Truth Table and Expected Outputs" << endl;

         for (int index = 0; index < numCases; ++index)
         {
            cout << trainData[index][0] << " & " << trainData[index][1] << " = " << trainAnswers[index] << " -> " << run(trainData[index]) << endl;
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
      cout << network->run(network->testData[index]) << endl << endl;
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
