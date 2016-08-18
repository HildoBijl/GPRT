# Source code for the Ph.D. thesis
Here you find the Matlab source code corresponding to the thesis "Gaussian Process Regression Techniques - With applications to wind turbines". The [thesis](http://hildobijl.com/Downloads/GPRT.pdf) itself can be downloaded from my [personal website](http://hildobijl.com/Research.php). It is written to introduce people to Gaussian process regression from an intuitive point of view.

## Contents

Every chapter in the thesis has its own folder. Just open the folder and you will find a Matlab script with the same name. For instance, `Chapter4.m`.

The experiments at the end of each chapter are generally done in a separate file. You will just have to guess which file name corresponds best to the experiment.

In general, just open a file, make sure that the folder it is in is set as Matlab's current working directory, run it and see what figures it comes up with.

## External toolboxes

I have used a few other toolboxes in the code. To make the code easily accessible, I have incorporated these toolboxes. (I know, not the best method, but it does prevent the code from breaking, as these toolboxes get updates.) If you work more with the code, you might want to download these toolboxes yourself though. They are the following.

* For exporting figures to files, the [export_fig](https://nl.mathworks.com/matlabcentral/fileexchange/23629-export-fig) package.
* For chapter 5: the NIGP toolbox. See the personal page of [Andrew McHutchon](http://mlg.eng.cam.ac.uk/?portfolio=andrew-mchutchon).
* For chapter 5: the SONIG toolbox. I developed this one myself, but it has [its own GitHub page](https://github.com/HildoBijl/SONIG).

All other support code (the Tools and the PitchPlunge system) have been coded by myself and have not been uploaded like this anywhere else before.

## Feedback

I am always curious where the stuff that I make winds up, so if you find this useful, send me a message. (You can find [contact information](http://hildobijl.com/Contact.php) on my [personal web page](http://hildobijl.com/).)

Oh, and if you find an error anywhere, then this should of course also be fixed so others aren't bothered by it. I will help fix those in any way I can.
