The realtime generated snapshots of the user search session should be saved in this folder. 

Each new snapshot (document) should be in a separate .npy file and have the following sparse format:

      doc = [(term_idx,freq),..] 

Please note that, at the moment, the term_idx should be the same as the term_idx in the original corpus (the corpus should not be updated based on the new doc)
Furthermore, please remove terms that are not in the current corpus.