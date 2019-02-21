import os


def os_system(argsv_ori):
    argsv = argsv_ori.split(',')
    data_file = argsv[0]
    golden_file = argsv[1]
    s = int(argsv[2])
    e = int(argsv[3])
    os.system('chmod 777 score')

    for i in range(s, e+1):
        cmd = './score %s/dic %s ../result/dev_result%s > tmp' % (data_file, golden_file, i)
        os.system(cmd)
        cmd = 'grep \'F MEASURE\' tmp '
        os.system(cmd)
        cmd = 'grep \'TOTAL TEST WORDS PRECISION\' tmp '
        os.system(cmd)
        cmd = 'grep \'TOTAL TRUE WORDS RECALL\' tmp '
        os.system(cmd)
        cmd = 'rm tmp'
        os.system(cmd)