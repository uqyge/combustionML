import os

def writeANNProperties(in_scaler,out_scaler,scaler):
    try:
        assert os.path.isdir('ANNProperties')
    except:
        os.mkdir('ANNProperties')

    ANNProperties = open('ANNProperties/ANNProperties_'+scaler, 'w')

    try:
        with open('ANNProperties/top', encoding='utf-8') as f:
            for line in f.readlines():
                ANNProperties.write(line)
    except:
        print('Include a header file named: top into ./ANNProperties!')

    # write which is the normalization scaler: MinMax or Standard
    ANNProperties.write('ANN_scaler         %s;\n' % scaler)

    if scaler == 'Standard':
        ANNProperties.write('\nin_scale\n')
        ANNProperties.write('{\n')
        for i in range(len(in_scaler.mean_)):
            mean_string = 'in_%i_mean      %.16f;\n' % (i + 1, in_scaler.mean_[i])
            var_string = 'in_%i_var        %.16f;\n' % (i + 1, in_scaler.scale_[i])
            ANNProperties.write(mean_string)
            ANNProperties.write(var_string)

        ANNProperties.write('}\n')
        ANNProperties.write('\nout_scale\n')
        ANNProperties.write('{\n')
        for i in range(len(out_scaler.mean_)):
            ANNProperties.write('out_%i_mean      %.16f;\n' % (i + 1, out_scaler.mean_[i]))
            ANNProperties.write('out_%i_var       %.16f;\n' % (i + 1, out_scaler.scale_[i]))
        ANNProperties.write('}\n')

        # write the number of species
        ANNProperties.write('nr_features         %i;\n' % len(out_scaler.mean_))

    elif scaler == 'MinMax':
        ANNProperties.write('\nin_scale\n')
        ANNProperties.write('{\n')
        for i in range(len(in_scaler.data_max_)):
            ANNProperties.write('in_%i_max       %.16f;\n' % (i + 1, in_scaler.data_max_[i]))
            ANNProperties.write('in_%i_min       %.16f;\n' % (i + 1, in_scaler.data_min_[i]))

        ANNProperties.write('}\n')

        ANNProperties.write('\nout_scale\n')
        ANNProperties.write('{\n')
        for i in range(len(out_scaler.data_max_)):
            ANNProperties.write('out_%i_max       %.16f;\n' % (i + 1, out_scaler.data_max_[i]))
            ANNProperties.write('out_%i_min       %.16f;\n' % (i + 1, out_scaler.data_min_[i]))
        ANNProperties.write('}\n')

        ANNProperties.write('\n')
        ANNProperties.write('range_min         %s;\n' % min(in_scaler.feature_range))
        ANNProperties.write('range_max         %s;\n' % max(in_scaler.feature_range))

        # write nr of species
        ANNProperties.write('nr_features         %i;\n' % len(out_scaler.data_max_))

    ANNProperties.write('\n// ************************************************************************* //')

    ANNProperties.close()

    print('\nANNProperties are written')