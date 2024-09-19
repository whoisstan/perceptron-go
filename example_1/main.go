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

// Perceptron struct with weights, bias, and learning rate
type Perceptron struct {
	Weights      []float64 // length of feature dimensionality
	Bias         float64
	LearningRate float64 // only needed for train
}

// Initialize the perceptron with random weights and bias
func NewPerceptron(inputSize int, learningRate float64) *Perceptron {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	weights := make([]float64, inputSize)
	for i := range weights {
		weights[i] = r.Float64() * 0.5
	}

	bias := rand.Float64() * 0.5
	return &Perceptron{
		Weights:      weights,
		Bias:         bias,
		LearningRate: learningRate,
	}
}

// Predict function for the perceptron
// w1x1 + w2x2 + ... + b
func (p *Perceptron) Predict(inputs []float64) int {
	var weightedSum float64
	for i := range inputs {
		weightedSum += inputs[i] * p.Weights[i]
	}
	weightedSum += p.Bias
	return step(weightedSum)
}

// Train the perceptron with early stopping and log weights/bias at each epoch
func (p *Perceptron) Train(inputs [][]float64, labels []int, maxEpochs int) [][]float64 {

	epochsData := [][]float64{}
	for epoch := 0; epoch < maxEpochs; epoch++ {
		misclassified := false
		for i := range inputs {
			prediction := p.Predict(inputs[i])
			error := labels[i] - prediction

			// If there is an error, update weights and bias
			if error != 0 {
				misclassified = true
				for j := range p.Weights {
					p.Weights[j] += p.LearningRate * float64(error) * inputs[i][j]
				}
				p.Bias += p.LearningRate * float64(error)
			}
		}

		// Log weights and bias after each epoch
		epochsData = append(epochsData, []float64{p.Weights[0], p.Weights[1], p.Bias})

		// Stop early if no misclassified points
		if !misclassified {
			fmt.Printf("Training stopped early at epoch %d\n", epoch+1)
			break
		}
	}
	return epochsData
}

func main() {
	dataset := [][]float64{
		{2, 3},
		{1, 1},
		{4, 5},
		{6, 7},
		{5, 1},
		{7, 3},
		{8, 7},
		{6, 5},
	}
	labels := []int{1, 0, 1, 1, 0, 0, 0, 0}

	// Initialize perceptron struct
	//
	perceptron := NewPerceptron(2, 0.1)

	// Train perceptron and collect weights/bias per epoch
	epochsData := perceptron.Train(dataset, labels, 100)

	// Plot decision boundary per epoch
	CreateOrEmptyOutFolder("epochs")
	for epoch, data := range epochsData {
		PlotEpoch("epochs", epoch, data[:2], data[2], dataset, labels)
	}
}
