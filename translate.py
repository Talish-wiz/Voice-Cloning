from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.argutils import print_args
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
import argparse
import torch
import sys

if __name__ == '__main__':
    ## Info & args
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # remove this, should be hardcoded elsewhere
    parser.add_argument("-e", "--enc_model_fpath", type=Path,default="encoder/saved_models/pretrained.pt",help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_dir", type=Path,default="synthesizer/saved_models/logs-pretrained/",help="Directory containing the synthesizer model")
    parser.add_argument("-v", "--voc_model_fpath", type=Path,default="vocoder/saved_models/pretrained/pretrained.pt",help="Path to a saved vocoder")
    # 

    parser.add_argument("--out", type=Path,default="output.wav", help="sets the output wav file")
    parser.add_argument("--textin", type=Path,help="sets the output wav file")
    parser.add_argument("--voicein", type=Path,default="input.wav",  help="sets the input wav file")

    args = parser.parse_args()
    print_args(args, parser)
        
    if args.textin:
        print("\nargs.textin: ", args.textin)
    else:
        print("\nNO VALUE: args.textin")   
        
    if args.voicein:
        print("\nargs.voicein: ", args.voicein)
    else:
        print("\nNO VALUE: args.voicein")       

    if args.out:
        print("\nargs.out: ", args.out)
    else:
        print("\nNO VALUE: args.out")         
    
    ## Print some environment information (for debugging purposes)
    print("Running a test of your configuration...\n")
    if not torch.cuda.is_available():
        print("Your PyTorch installation is not configured to use CUDA. If you have a GPU ready "
              "for deep learning, ensure that the drivers are properly installed, and that your "
              "CUDA version matches your PyTorch installation. CPU-only inference is currently "
              "not supported.", file=sys.stderr)
        quit(-1)
    device_id = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device_id)
    print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
          "%.1fGb total memory.\n" % 
          (torch.cuda.device_count(),
           device_id,
           gpu_properties.name,
           gpu_properties.major,
           gpu_properties.minor,
           gpu_properties.total_memory / 1e9))
    
    
    ## Load the models one by one.
    print("Preparing the encoder, the synthesizer and the vocoder...")
    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_dir.joinpath("taco_pretrained"))
    vocoder.load_model(args.voc_model_fpath)


    # ********************************
    if args.voicein:
        print("args.voicein: ", args.voicein)
        in_fpath = args.voicein
    else:
        # Get the reference audio filepath
        message = "Reference voice: enter an audio filepath of a voice to be cloned (mp3, wav, m4a, flac, ...):\n"
        in_fpath = Path(input(message).replace("\"", "").replace("\'", ""))

    print("in_fpath: ", in_fpath)
    

    embeds = []
    embeds.append(encoder.embed_utterance(encoder.preprocess_wav("input.wav")))



    print("Interactive generation loop")

    try:
        ## Generating the spectrogram
        if args.textin:
            text = str(args.textin)
        else:
            text = input("Write a sentence (+-20 words) to be synthesized:\n")

        
        # The synthesizer works in batch, so you need to put your data in a list or numpy array
        texts = [text]
        #embeds = [embed]
        # If you know what the attention layer alignments are, you can retrieve them here by
        # passing return_alignments=True
        specs = synthesizer.synthesize_spectrograms(texts, embeds)
        spec = specs[0]
        print("Created the mel spectrogram")
        
        
        ## Generating the waveform
        print("Synthesizing the waveform:")
        # Synthesizing the waveform is fairly straightforward. Remember that the longer the
        # spectrogram, the more time-efficient the vocoder.
        generated_wav = vocoder.infer_waveform(spec)
        
        ## Post-generation
        # There's a bug with sounddevice that makes the audio cut one second earlier, so we
        # pad it.
        generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
        
        # Save it on the disk
        if args.out:
            fpath = args.out
        else:
            fpath = "output.wav"

        print("generated_wav.dtype: ", generated_wav.dtype)

        librosa.output.write_wav(fpath, generated_wav.astype(np.float32), synthesizer.sample_rate)

        print("\nSaved output as %s\n\n" % fpath)
        
        
    except Exception as e:
        print("Caught exception: %s" % repr(e))
        print("Restarting\n")