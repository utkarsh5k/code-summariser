import sys
import numpy as np
import cPickle

from copy_conv_rec_learner import ConvolutionalCopyAttentionalRecurrentLearner

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print "Usage inputModel inputFile outputPkl"
        sys.exit(-1)

    model = ConvolutionalCopyAttentionalRecurrentLearner.load(sys.argv[1])
    data, original_names = model.naming_data.validated_rec_copy_conv_data(sys.argv[2], model.padding_size)
    name_targets, code_sentences, code, target_is_unk, copy_vectors = data

    context = np.array([model.naming_data.all_tokens_dictionary.is_id_or_is_unknown(model.naming_data.SUBTOKEN_START)], dtype=np.int32)

    attention_features = []
    for i in xrange(len(code)):
        attention_feat = model.model.attention_features(code_sentences[i], context)
        attention_features.append((original_names[i], code[i], attention_feat))

    with open(sys.argv[3], 'wb') as f:
        cPickle.dump(attention_features, f)