To generate sentences:
- Download main_generate.py
- Download model_generate
- Make sure main_generate.py and model_generate are in the same directory, then just run main_generate.py 
either in an IDE or straight from the command line. Output should be 10 generated lines, each of 100 characters.

Important Notes:
- All of the code used in part 3 of this homework is included. All of the functions and classes are there, but 
some of the code (network training loop, plots, perplexity calculations, etc.) are commented out, so that the only 
thing that is output when running the file are the generated sentences. The rest of the code can be run if necessary 
just by uncommenting it, and then ensuring the necessary text files, mentioned in the writeup, are included in the 
same directory as main_generate.py.
- The code was commented out to make generating sentences easier. It loads in the pretrained model and then generates 
using that, as training the model takes a few minutes. 
- To train the model with a different text file, change the variable text_string at the beginning of the training loop 
to call get_text_string with a different read_location. Change the location to the desired text file, and it should work.
