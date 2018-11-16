import os

def writeANNproperties(in_scaler,out_scaler):
    try:
        assert os.path.isdir('ANNProperties')
    except:
        os.mkdir('ANNProperties')

    ANNProperties = open('ANNProperties/ANNProperties', 'w')

    with open('ANNProperties/top', encoding='utf-8') as f:
        for line in f.readlines():
            ANNProperties.write(line)

    ANNProperties.write('in_scale\n')
    ANNProperties.write('{\n')
    for i in range(len(in_scaler.mean_)):
        mean_string = 'in_%i_mean      %f;\n' % (i + 1, in_scaler.mean_[i])
        var_string = 'in_%i_var       %f;\n' % (i + 1, in_scaler.scale_[i])
        ANNProperties.write(mean_string)
        ANNProperties.write(var_string)

    ANNProperties.write('}\n')
    ANNProperties.write('\nout_scale\n')
    ANNProperties.write('{\n')
    for i in range(len(out_scaler.mean_)):
        ANNProperties.write('out_%i_mean      %f;\n' % (i + 1, out_scaler.mean_[i]))
        ANNProperties.write('out_%i_var       %f;\n' % (i + 1, out_scaler.scale_[i]))
    ANNProperties.write('}\n')
    ANNProperties.write('\n// ************************************************************************* //')

    ANNProperties.close()

    print('ANNProperties are written')