# encoding=utf-8

import numpy as np
# from scipy import signal
# import matplotlib.pyplot as plt
# from tfridge_tracking import ridge_tracking as rt
# from pykalman import KalmanFilter
# import cmath

'''
基本参数设置
'''
ADC_Samples = 256
Nfft1 = 256
Nchirp = 128
Nfft2 = 128
NTx = 3
NRx = 4
C = 3e8
startFreq = 60.05e9
sampleRate = 10e6
freqSlope = 130.029e12

'''
波束参数设置
'''
lambdaL = C / startFreq
d = lambdaL / 2

print(lambdaL, d)
'''
推算参数
'''
Tchirp_ramp = 30e-6
Tchirp = (Tchirp_ramp + 100e-6) * 3  # 循环Chirp总时间, TDM-MIMO模式


def LoadSinglefile(fileName, numAdcSamples, numRX, numTx):
    '''
    Input:
        fileName: 原始bin文件
        numAdcSamples: ADC采样的数量
        numRX: 接收天线的数量
        numTX: 发射天线的数量
    Output:
        LVDS: 12, numframe * numChirp * numAdcSample 12根虚拟天线 , frame的数量,Chirp的数量,Adc样本的数量
    '''
    adc_data = np.fromfile(fileName, dtype=np.int16)
    filesize = len(adc_data)
    adc_data = adc_data.reshape(-1, 4)
    LVDS = np.zeros(shape=[filesize // 4, 2], dtype=np.complex64)
    LVDS[:, 0] = adc_data[:, 0] + 1.j * adc_data[:, 2]
    LVDS[:, 1] = adc_data[:, 1] + 1.j * adc_data[:, 3]
    LVDS = LVDS.reshape(-1)
    numChirps = filesize // (2 * numAdcSamples * numRX)
    # Chirp(一共收发了这些个Chirps), numTx*numSamples*numRx
    LVDS = LVDS.reshape(numChirps // numTx, numAdcSamples * numRX * numTx)
    # 每个Chirp中分别包含了12个虚拟收发天线之间的数据 Chirps,12根虚拟天线接收的ADCSamples
    LVDS = LVDS.reshape(-1, numTx * numRX, numAdcSamples).swapaxes(1, 0)
    LVDS = LVDS.reshape(numTx * numRX, -1)
    return LVDS


def processSinglefile(fileName, numAdcSamples, numRX, numTx, Nfft1, Nfft2, numChirp, numFrame):
    adcDataList = []

    framecount = 0
    for ii in range(0, 3600, 600):
        fileNamecur = fileName + "_" + str(ii) + "_" + str(ii + 600) + ".bin"
        print("Cur FileName: ", fileNamecur)
        adcData = LoadSinglefile(fileNamecur, numAdcSamples, numRX, numTx)
        for frameIdx in range(numFrame):
            # print("cur FrameIdx: ",frameIdx)
            # 逐帧处理 Nfft2是一帧的Chirp数量,Nfft1是AdcSample的数量
            # NRx*NTx, Chirp*Nsample
            adcDataIn = adcData[:, frameIdx * Nfft2 * Nfft1:(frameIdx + 1) * Nfft2 * Nfft1]
            adcDataIn = adcDataIn.reshape(numRX * numTx, Nfft2, Nfft1)  # numRx*numTx,Nfft2, NumSample
            adcDataout = np.fft.fft2(adcDataIn)  # numRx*numTx, Nfft2, Nfft1
            adcDataoutshift = np.fft.fftshift(adcDataout, axes=1)
            adcDataList.append(adcDataoutshift)
            framecount += 1


if __name__ == "__main__":

    for i in range(22, 24):
        if i == 6:
            continue
        print("Current Person: Person" + str(i))
        processSinglefile("../L316RadarData/Plant/Person" + str(i), numAdcSamples=256, numRX=4, numTx=3, Nfft1=256,
                          Nfft2=128, numChirp=128, numFrame=600)
    # for i in range(1,24):
    #   if i == 6:
    #        continue
    #    print("Current Person: Person"+str(i))
    #    processSinglefile("../L304RadarData/Poster/Person"+str(i),numAdcSamples=256,numRX=4,numTx=3,Nfft1=256,Nfft2=128,numChirp=128,numFrame=600)   
    # for i in range(22,24):
    #    if i == 6:
    #        continue
    #    print("Current Person: Person"+str(i))
    #    processSinglefile("../L304RadarData/Empty/Person"+str(i),numAdcSamples=256,numRX=4,numTx=3,Nfft1=256,Nfft2=128,numChirp=128,numFrame=600)
    # for i in range(1,24):
    #    if i == 6:
    #        continue
    #    print("Current Person: Person"+str(i))
    #    processSinglefile("../L304RadarData/HatRack/Person"+str(i),numAdcSamples=256,numRX=4,numTx=3,Nfft1=256,Nfft2=128,numChirp=128,numFrame=600)

    # ridgeindex2 = processSinglefile("../L316RadarData/Plant/Person"+str(2)+"_0_600.bin",numAdcSamples=256,numRX=4,numTx=3,Nfft1=256,Nfft2=128,numChirp=128,numFrame=600)
    # a = np.array([1,2,3,4,5])
    # print(a.shape)
    # print((105-Nchirp//2) * lambdaL / Tchirp / Nchirp / 2)
