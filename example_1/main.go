package main

import (
	"fmt"
	"math/rand"
	"time"
)

// Step function
//
//	f(x) = {
//				0, x < 0
//				1, x >= 0
//	}
func step(x float64) int {
	if x >= 0 {
		return 1
	}
	return 0
}

// Perceptron Model struct with weights, bias, and learning rate
type Model struct {
	Weights []float64 // length of feature dimensionality
	Bias    float64
}

// Initialize the model with random weights and bias
func SetupModel(inputSize int) *Model {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	weights := make([]float64, inputSize)
	for i := range weights {
		weights[i] = r.Float64() * 0.5
	}

	bias := rand.Float64() * 0.5
	return &Model{
		Weights: weights,
		Bias:    bias,
	}
}

// Predict function for the perceptron
// w1x1 + w2x2 + ... + b
func (m *Model) Predict(inputs []float64) int {
	var weightedSum float64
	for i := range inputs {
		weightedSum += inputs[i] * m.Weights[i]
	}
	weightedSum += m.Bias
	return step(weightedSum)
}

// Train the model with early stopping and log weights/bias at each epoch
func (m *Model) Train(inputs [][]float64, labels []int, maxEpochs int, learningRate float64) []Model {

	epochsData := []Model{}
	for epoch := 0; epoch < maxEpochs; epoch++ {
		misclassified := false
		for i := range inputs {
			prediction := m.Predict(inputs[i])
			difference := labels[i] - prediction

			// If there is a difference, update weights and bias using the learningRate
			if difference != 0 {
				misclassified = true
				for j := range m.Weights {
					m.Weights[j] += learningRate * float64(difference) * inputs[i][j]
				}
				m.Bias += learningRate * float64(difference)
			}
		}

		// Stop early if no misclassified points
		if !misclassified {
			fmt.Printf("Training stopped early at epoch %d\n", epoch+1)
			break
		} else {
			// Create a deep copy of m.Weights
			weightsCopy := make([]float64, len(m.Weights))
			copy(weightsCopy, m.Weights)
			epochsData = append(epochsData, Model{Weights: weightsCopy, Bias: m.Bias})
		}
	}
	return epochsData
}

func main() {

	features := [][]float64{
		{2, 3},
		{1, 1},
		{4, 5},
		{6, 7},
		{5, 1},
		{7, 3},
		{8, 7},
		{6, 5},
	}
	//correspond to the feature rows
	labels := []int{1, 0, 1, 1, 0, 0, 0, 0}

	// Initialize perceptron struct with the dimensionality of the features
	model := SetupModel(len(features[0]))

	// Train model and collect weights/bias per epoch
	epochsData := model.Train(features, labels, 100, 0.1)

	PlotMultiEpochs(epochsData, features, labels, 4, "visual.png")
}
