package main

import (
	"fmt"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"
	"image/color"
	"image/png"
	"log"
	"math"
	"os"
	"path/filepath"
)

// Function to create or recreate the folder
func CreateOrEmptyOutFolder(folder string) {

	// Check if folder exists
	if _, err := os.Stat(folder); !os.IsNotExist(err) {
		// If it exists, remove it
		err := os.RemoveAll(folder)
		if err != nil {
			panic(err)
		}
	}

	// Create the folder
	err := os.Mkdir(folder, 0755)
	if err != nil {
		panic(err)
	}
}

// Function to plot the points and hyperplane using image/color
func PlotEpoch(folder string, index int, epochModel Model, inputs [][]float64, labels []int) {
	// Ensure the folder is created or recreated

	p := plot.New()

	p.Title.Text = fmt.Sprintf("Epoch %d - Perceptron Decision Boundary", index+1)
	p.X.Label.Text = "x1"
	p.Y.Label.Text = "x2"

	// Plot the OR dataset points
	pts := make(plotter.XYs, len(inputs))
	ptsRed := make(plotter.XYs, 0)
	ptsBlue := make(plotter.XYs, 0)

	for i := range inputs {
		pts[i].X = inputs[i][0]
		pts[i].Y = inputs[i][1]
		if labels[i] == 1 {
			ptsBlue = append(ptsBlue, pts[i])
		} else {
			ptsRed = append(ptsRed, pts[i])
		}
	}

	// Create scatter plot for red and blue points
	scatterRed, _ := plotter.NewScatter(ptsRed)
	scatterRed.GlyphStyle.Color = color.RGBA{R: 255, A: 255} // Red for class 0

	scatterBlue, _ := plotter.NewScatter(ptsBlue)
	scatterBlue.GlyphStyle.Color = color.RGBA{B: 255, A: 255} // Blue for class 1

	// Plot the decision boundary (hyperplane)
	line := plotter.NewFunction(func(x float64) float64 {
		return -(epochModel.Weights[0]*x + epochModel.Bias) / epochModel.Weights[1]
	})
	log.Print(line)
	line.Color = color.RGBA{G: 255, A: 255} // Green for hyperplane

	// Add scatter plot and line
	p.Add(scatterRed, scatterBlue, line)

	// Save the plot to the "plots" folder
	filePath := filepath.Join(folder, fmt.Sprintf("epoch_%03d.png", index+1))
	if err := p.Save(4*vg.Inch, 4*vg.Inch, filePath); err != nil {
		panic(err)
	}
}

// Function to plot multiple epochs on one large plot canvas using `gonum/plot`
func PlotMultiEpochs(epochsData []Model, inputs [][]float64, labels []int, cols int, fileName string) error {
	// Determine the size of each individual plot (sub-plot)
	plotWidth := 4 * vg.Inch
	plotHeight := 4 * vg.Inch

	// Calculate the number of rows
	totalEpochs := len(epochsData)
	rows := int(math.Ceil(float64(totalEpochs) / float64(cols)))

	// Create a blank canvas to hold all subplots
	totalWidth := vg.Length(cols) * plotWidth
	totalHeight := vg.Length(rows) * plotHeight
	canvas := vgimg.New(totalWidth, totalHeight)
	dc := draw.New(canvas)

	// Create tiles to divide the canvas
	tiles := draw.Tiles{
		Rows:      rows,
		Cols:      cols,
		PadX:      vg.Inch / 5,
		PadY:      vg.Inch / 5,
		PadRight:  vg.Inch / 5,
		PadTop:    vg.Inch / 5,
		PadLeft:   vg.Inch / 5,
		PadBottom: vg.Inch / 5,
	}

	// Loop through each epoch to create a sub-plot
	for i := 0; i < totalEpochs; i++ {
		// Calculate row and column for this subplot
		row := i / cols
		col := i % cols

		// Create a new sub-plot
		subPlot := plot.New()

		subPlot.Title.Text = fmt.Sprintf("Epoch %d", i+1)
		subPlot.X.Label.Text = "x1"
		subPlot.Y.Label.Text = "x2"

		// Plot the dataset points
		ptsRed := make(plotter.XYs, 0)
		ptsBlue := make(plotter.XYs, 0)

		for j := range inputs {
			x := inputs[j][0]
			y := inputs[j][1]
			if labels[j] == 1 {
				ptsBlue = append(ptsBlue, plotter.XY{X: x, Y: y})
			} else {
				ptsRed = append(ptsRed, plotter.XY{X: x, Y: y})
			}
		}

		// Create scatter plot for red and blue points
		scatterRed, _ := plotter.NewScatter(ptsRed)
		scatterRed.GlyphStyle.Color = color.RGBA{R: 255, A: 255} // Red for class 0

		scatterBlue, _ := plotter.NewScatter(ptsBlue)
		scatterBlue.GlyphStyle.Color = color.RGBA{B: 255, A: 255} // Blue for class 1

		// Plot the decision boundary (hyperplane)
		line := plotter.NewFunction(func(x float64) float64 {
			return -(epochsData[i].Weights[0]*x + epochsData[i].Bias) / epochsData[i].Weights[1]
		})
		line.Color = color.RGBA{G: 255, A: 255} // Green for hyperplane

		// Add scatter plot and line to the sub-plot
		subPlot.Add(scatterRed, scatterBlue, line)

		// Optionally set axis limits for consistency
		//subPlot.X.Min = -10
		//subPlot.X.Max = 10
		//subPlot.Y.Min = -10
		//subPlot.Y.Max = 10

		canv := tiles.At(dc, col, row)
		subPlot.Draw(canv)
	}

	// Save the combined plot to a PNG file
	w, err := os.Create(fileName)
	if err != nil {
		return err
	}
	defer w.Close()

	// Encode the canvas image to PNG and write to the file
	if err := png.Encode(w, canvas.Image()); err != nil {
		return err
	}

	return nil
}
