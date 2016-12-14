from subprocess import check_call
import os
import sys

def main(job_id, params):
    learning_rate  = 2 ** params['learning_rate_log'][0]
    ###dropout        = params['dropout'][0]
    dropout        = 0.5
    rnn_num_layers = params['rnn_num_layers'][0]
    rnn_num_units  = params['rnn_num_units'][0]
    cnn_num_layers = params['cnn_num_layers'][0]
    cnn_batch_norm = params['cnn_batch_norm'][0]
    cnn_features_m = params['cnn_features_mult'][0]
    cnn_num_features = [ str(cnn_features_m * (2 ** i)) \
                         for i in xrange(cnn_num_layers) ]
    cnn_maxpool_pending = params['cnn_maxpool']
    cnn_maxpool_size = []
    for i in xrange(cnn_num_layers):
        kw = 2 if cnn_maxpool_pending[0] > 0 else 1
        kh = 2 if cnn_maxpool_pending[1] > 0 else 1
        if kw == 1 and kh == 1:
            cnn_maxpool_size.append('0')
        else:
            cnn_maxpool_size.append('%d,%d' % (kw, kh))
        cnn_maxpool_pending[0] -= 1
        cnn_maxpool_pending[1] -= 1

    # Change to main directory
    cwd = os.getcwd()
    os.chdir('..')

    # Create model
    create_model_call = [
        '../../laia-create-model',
        '--cnn_type', 'leakyrelu',
        '--cnn_kernel_size', '3',
        '--cnn_num_features'] + cnn_num_features + [
        '--cnn_maxpool_size'] + cnn_maxpool_size + [
        '--cnn_batch_norm', cnn_batch_norm,
        '--cnn_dropout', '0',
        '--rnn_type', 'blstm',
        '--rnn_num_layers', str(rnn_num_layers),
        '--rnn_num_units', str(rnn_num_units),
        '--rnn_dropout', str(dropout),
        '--linear_dropout', str(dropout),
        '--log_level', 'info',
        '1', '64', '79', 'spearmint/model_%d.t7' % job_id
    ]
    print >> sys.stderr, ' '.join(create_model_call)
    check_call(create_model_call)

    # Run training
    train_ctc_call = [
        '../../laia-train-ctc',
        '--use_distortions', 'true',
        '--batch_size', '32',
        '--progress_table_output', 'spearmint/train_%d.dat' % job_id,
        '--early_stop_epochs', '25',
        '--early_stop_threshold', '0.05',
        '--max_epochs', '300',
        '--learning_rate', str(learning_rate),
        '--log_level', 'info',
        'spearmint/model_%d.t7' % job_id,
        'data/htr/lang/char/symbs.txt',
        'data/htr/tr.lst', 'data/htr/lang/char/tr.txt',
        'data/htr/va.lst', 'data/htr/lang/char/va.txt'
    ]
    print >> sys.stderr, ' '.join(train_ctc_call)
    check_call(train_ctc_call)

    # Get validation CER
    f = open('spearmint/train_%d.dat' % job_id, 'r')
    min_cer = 100000;
    for line in f:
        line = line.split()
        if line[0][0] == '#': continue
        cer = float(line[4])
        min_cer = min(min_cer, cer)
    f.close()

    # Remove model file, to save space
    os.remove('spearmint/model_%d.t7' % job_id)

    # Return to spearmint directory
    os.chdir(cwd)

    return min_cer
