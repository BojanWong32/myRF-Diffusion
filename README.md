## 学习文档

本文是对https://github.com/mobicom24/RF-Diffusion 和 http://tns.thss.tsinghua.edu.cn/widar3.0/ 的学习文档，个人使用

#### 1月19日，整理一下之前的学习成果

##### 1.测试代码可行性

数据解析https://github.com/zzh606/wifi-csi

该扩散模型所应用的数据集是处理后的Widar3.0数据集，分为feature和condition两部分。其中feature指特征矩阵，是一个t\*90的复数双精度矩阵，t表示时间长度，90由30\*3组成，3是天线数量，30是每根天线每个时间点发送的包数。condition表示标签，分别表示“房间-手势-躯干位置-面部朝向-接收器ID-userID”（1\*6\*5\*5\*20\*6）。

<img src="image/image-20240120124212107.png" style="zoom:33%;" />

使用代码提供的模型和测试数据，得到平均SSIM：

<img src="image/image-20240120124512276.png" style="zoom:33%;" />

原数据与生成数据的多普勒偏移对比：

<img src="image/image-20240120124633326.png" style="zoom:50%;" />

##### 2.准备数据，训练模型

对数据集进行处理，将CSI数据处理成能够传入神经网络的格式

```
data_path = 'Data\20181109\user1';
output_path = 'Output\20181109\user1';
roomID = 1;
file_list = dir(fullfile(data_path, '*.dat'));
for i = 1:length(file_list)
    name = file_list(i).name; % userID-手势-躯干位置-面部朝向-重复次数-Rx接收器id
    newName = strrep(name, '.dat', '.mat');
    file_path = fullfile(data_path, name);
    [feature, timestamp] = csi_get_all(file_path);
    disp(['Processed file: ' file_list(i).name]);

    name = strrep(name, 'user', '');  % 去除前缀 "user"
    name = strrep(name, 'r', '');  % 去除后缀 "r"
    name = strrep(name, '.dat', '');
    nameArray = strsplit(name, '-');
    cond = str2double(nameArray); 

    cond(7) = cond(1);
    cond(5) = [];
    cond(1) = roomID; % 房间-手势-躯干位置-面部朝向-接收器ID-userID
    save(fullfile(output_path, newName), 'feature', 'cond');

end
```

最终得到18000条数据，使用提供的代码和自己处理得到的数据进行训练，得到结果与上述相似。

#### 1月20日，计划使用Widar3.0的神经网络测试扩散模型对数据的增强作用

##### 1.BVP（Body-coordinate Velocity Profile）

对于该神经网络，需要再次将数据格式转为20\*20\*t的BVP形式，即：

<img src="image/bvp_domain1.gif" style="zoom: 50%;" />

该数据是由六个接收器的CSI数据计算而来
学习从CSI到BVP的生成代码

##### 2.选择数据，梳理思路

无线感知系统是根据BVP数据来识别该CSI对应的手势。由于之前训练的扩散模型只选择了两个手势作为样本，现在将它们应用到该神经网络中，在分类数量上是没有什么说服力的。

（1）所以首先，我计划使用包含6种手势的数据集来训练一个扩散模型。

（2）然后，将上面提到的处理好的真实数据生成相同数量的合成数据。

（3）再然后，将这些CSI矩阵转化为BVP形式。

（4）最后，测试合成数据对数据集的增强效果，比较方便的一点是，在原数据集中对每个相同的条件进行了二十次重复测试，所以，计划进行六次模型训练，分别只使用真实数据，使用真实数据加二十分之四合成数据，使用真实数据加二十分之八合成数据，……，使用全部真实数据和合成数据，比较几个模型的精确度印证合成数据对无线感知系统的增强。

#### 1月21日

##### 1.开始重新训练扩散模型，大概需要20个小时，期间看一下BVP的生成代码

```
for mo_sel = 1:total_mo
    for pos_sel = 1:total_pos
        for ori_sel = 1:total_ori
            for ges_sel = 1:total_ges
                spfx_ges = [dpth_people, '-', num2str(mo_sel), '-', num2str(pos_sel),...
                    '-', num2str(ori_sel), '-', num2str(ges_sel)];
                if mo_sel == start_index(1) && pos_sel == start_index(2) &&...
                        ori_sel == start_index(3) && ges_sel == start_index(4)
                    start_index_met = 1;
                end
                if start_index_met == 1
                    disp(['Running ', spfx_ges])
                    try
                        DVM_main;
                    catch err
                        disp(['Exception Occured' err.message]);
                        fprintf(exception_fid, '%s\n', spfx_ges);
                        fprintf(exception_fid, '%s\n', err.message);
                    	continue;
                    end
                else
                    disp(['Skipping ', spfx_ges])
                end
            end
        end
    end
end
```

依次遍历手势、位置等来组合文件名，随后执行主要代码块DVM_main.m

```
[doppler_spectrum, freq_bin] = get_doppler_spectrum([dpth_ges, spfx_ges],...
                    rx_cnt, rx_acnt, 'stft');
```

首先执行get_doppler_spectrum函数，用于分析无线信号多普勒效应，检测物体的运动速度和方向。输入为数据地址，接收器数量=6，每个接收器天线数=3，时频分析方法stft（短时傅里叶变换）。输出为多普勒频谱矩阵和频率索引向量。

```
% For Each Segment Do Mapping
doppler_spectrum_max = max(max(max(doppler_spectrum,[],2),[],3));
U_bound = repmat(doppler_spectrum_max, M, M);
A = get_A_matrix(torso_pos(pos_sel,:), Tx_pos, Rx_pos, rx_cnt);
VDM = permute(get_velocity2doppler_mapping_matrix(A, wave_length,...
    velocity_bin, freq_bin, rx_cnt), [2,3,1,4]);    % 20*20*rx_cnt*121
% CastM = get_CastM_matrix(A, wave_length, velocity_bin, freq_bin);
```

生成表示速度到多普勒频谱映射关系的矩阵VDM

```
seg_number = floor(size(doppler_spectrum, 3)/seg_length);
    doppler_spectrum_ges = doppler_spectrum;
    velocity_spectrum = zeros(M, M, seg_number);
    parfor ii = 1:seg_number
        % Set-up fmincon Input
        doppler_spectrum_seg = doppler_spectrum_ges(:,:,...
            (ii - 1)*seg_length+1 : ii*seg_length);
        doppler_spectrum_seg_tgt = mean(doppler_spectrum_seg, 3);
        
        % Normalization Between Receivers(Compensate Path-Loss)
        for jj = 2:size(doppler_spectrum_seg_tgt,1)
            if any(doppler_spectrum_seg_tgt(jj,:))
                doppler_spectrum_seg_tgt(jj,:) = doppler_spectrum_seg_tgt(jj,:)...
                    * sum(doppler_spectrum_seg_tgt(1,:))/sum(doppler_spectrum_seg_tgt(jj,:));
            end
        end

        % Apply fmincon Solver
        [P,fval,exitFlag,output] = fmincon(...
            @(P)DVM_target_func(P, VDM, lambda, doppler_spectrum_seg_tgt, size(doppler_spectrum_seg_tgt,1), norm),...
            zeros(M,M),...  % Initial Value
            [],[],...       % Linear Inequality Constraints
            [],[],...       % Linear Equality Constraints
            zeros(M,M),...  % Lower Bound
            U_bound,...     % Upper Bound
            [],... % @(P)DVM_nonlinear_func(P, CastM),...    % Non-linear Constraints
            optimoptions('fmincon','Algorithm','sqp',...
            'MaxFunctionEvaluations', MaxFunctionEvaluations));	% Options
        velocity_spectrum(:,:,ii) = P;
        exitFlag
    end
    
    % Rotate Velocity Spectrum According to Orientation
    velocity_spectrum_ro = get_rotated_spectrum(velocity_spectrum, torso_ori(ori_sel));
```

将多普勒频谱图根据时间分段，取平均值，接收器间进行归一化处理以补偿路径损耗，然后通过使用fmincon函数得到优化后的速度谱P。

##### 2.更新，考虑到训练和处理18000条数据所需时间过多，将数据集更改为原来的四分之一，即只选择重复测量五次的数据。
##### 3.新模型训练好了，SSIM还可以（0.86），但是需要重写一下扩散模型和matlab脚本的输入输出来合成BVP。
首先，在原模型中为了保证随机性，对数据集进行了混淆也没有记录原文件名，但生成BVP数据需要相同条件相同重复次数的六个接收器的数据计算生成；

其次，原数据为t\*90（t为时间且不固定），但合成数据均为512\*90的矩阵，所以虽然数据集中提供了部分原数据的BVP，但仍需重新计算将其更改为512\*90的矩阵

#### 1月22日
##### 1.遇到报错RuntimeError: stack expects a non-empty TensorList
更改Dataloader的drop_last和num_workers，均无效，尝试几次发现每次卡在同样的一条数据，查看该文件，发现文件大小明显不对，其他文件都在2000kb左右，而该文件只有200kb，查看矩阵发现该矩阵只有219*90，小于我们的最低标准512，再查看其他相同条件矩阵，应该是采集数据时发生错误，将该组数据删除。
##### 2.转化BVP数据
将749条512\*90的真实数据和749条512\*90的合成数据
```
function [cfr_array, timestamp] = csi_get_all(filename)


csi_trace = 512;
timestamp = zeros(length(csi_trace), 1);
% cfr_array = zeros(length(csi_trace), 90);

data = load(filename);
pred = data.pred;

pred = double(pred);

cfr_array = reshape(pred, [512, 90]);
```
##### 3.数据转化太慢
先处理好真实数据的BVP，用时7个小时

先测试只含有真实数据的情况，749条分为674条训练数据，75条测试数据，共计六种手势。

准确度随着轮数增加而提高，说明该神经网络代码和数据应该问题不大。

但是，由于数据量过少，即使提高轮次，最终的测试集准确度仍不高，测试五次，结果分别为[0.4933 0.5066 0.52 0.4933 0.5466]，平均测试集准确度为0.5119。

#### 1月23日
复现结果不理想，随着合成数据的增加，准确度反而下降

<img src="image/image_old.png" style="zoom:50%;" />

可能失败原因：

1.数据量不足，使用3000条20*20*t真实BVP数据时，准确率能达到百分之九十多，而现在即使只使用真实数据也只有百分之五十，需下一步使用更多数据

2.BVP转化代码改写错误

3.扩散模型代码改写错误

4.时间维度对无线感知也有影响，不能只使用512时间长度的数据


这张图是两张相同标签的六个天线的多普勒频移图，左合成，右真实

<img src="image/8df93d48701269deea239633aa9e734.png" style="zoom:50%;" />


##### 1.是不是BVP生成代码的问题
先用t*90的CSI跑一遍该代码

生成的BVP与标准BVP比较，矩阵大小相同说明是同样生成的，但内容完全不同

<img src="image/696b810981c072d41e7f720ca0df73e.png" style="zoom:50%;" />


改用dat文件从头生成BVP，也不相同，难道是我之前改什么地方，改错参数了吗。

重新解压生成代码，还是照旧，先不管。

##### 2.还是先处理数据吧，看看18000条数据的结果如何

#### 1月24日

18000条CSI数据仍然只有0.5的准确度，可能是时间维度对无线感知也有影响，尝试保留时间维度，合成20*20*t的BVP数据

#### 1月31日，更新

使用时间长度为t的真实CSI矩阵训练无线感知系统，最终准确度有90%，算是达到了标准，看来矩阵长度确实也应该算是重要特征之一。

#### 2月1日

```
def save(out_dir, data, cond, batch, index=0, file_name='xxx'):
    os.makedirs(out_dir, exist_ok=True)
    file_name = os.path.join(out_dir, file_name)
    time = times[batch]
    # print(time)
    # print(data.shape)

    data = torch.view_as_real(data).permute(0, 2, 3, 1)
    # print(data.shape)
    data = F.interpolate(data, (2, time), mode='nearest-exact')

    # print(data.shape)
    data = data.permute(0, 3, 1, 2)
    data = data.contiguous()
    # print(data.shape)
    data = torch.view_as_complex(data)
    # print(data.shape)

    mat_data = {
        'pred': data.numpy(),
        'cond': cond.numpy()
    }
    # print(file_name)
    scio.savemat(file_name, mat_data)
```
修改save函数，使其根据原CSI矩阵时间长度，将合成数据变成相应大小。

#### 2月3日

<img src="image/image.png" style="zoom:50%;" />
又复现失败了，感觉是这个Widar3数据集的无线感知神经网络太复杂，换一个简单的再试试

#### 2月14日

学习https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark

#### 3月25日

<img src="image/40a2d72fd9abcbd2222563436edfd20.png" style="zoom:50%;" />

#### 4月1日

尝试使用https://github.com/yongsen/SignFi

#### 4月2日

将512\*90\*6变成512\*180
```
path = 'D:\RFDiffusion\RF-Diffusion-main\dataset\wifi\old2';
new_path = 'D:\RFDiffusion\RF-Diffusion-main\dataset\wifi\merge';

files = dir(path);

files_per_group = 6;

for i = 3:files_per_group:numel(files)
    old_file_name = files(i).name;
    new_file_name = regexprep(old_file_name, '-r\d+', '');
    new_file_path = fullfile(new_path, new_file_name);
    
    data = zeros(512,180);
    start_col = 1;

    for j = i:min(i + files_per_group - 1, numel(files))
        file_name = files(j).name;
        file_path = fullfile(path, file_name);

        current_file = load(file_path);
        if j == i
            cond = current_file.cond; 
        end

        end_col = start_col + 29;
        matrix =  squeeze(current_file.pred);
        data(:,start_col:end_col) = matrix(1:512,1:30);
        start_col = end_col + 1;
    end
    
    save(new_file_path, 'data','cond');
end
```
修改参数，input_dim=180，extra_dim=180,signal_diffusion=False,batch_size=4

#### 4月3日

训练大约20h，训练了50轮，loss约为1e-2数量级。

再次修改BVP合成代码，512\*90\*6 改512\*180

首先是原先的-rx变成合集的第x个30列

```
function [cfr_array, timestamp] = csi_get_all(filename)

csi_trace = 512;
timestamp = zeros(length(csi_trace), 1);
% cfr_array = zeros(length(csi_trace), 90);

pattern = '-r(\d+)';
num = regexp(filename, pattern, 'match', 'once');
filename = regexprep(filename, num, '');
value = str2double(regexprep(num, '-r', ''));

disp(value);

matrix = load(filename);
data = matrix.data(:,(value-1)*30+1 : value*30);

data = double(data);

cfr_array = squeeze(data);
```
数组越界，调试，发现问题所在，尝试改动

```
% Conj Mult
conj_mult = csi_data_adj .* conj(csi_data_ref_adj);
%conj_mult = [conj_mult(:,1:30*(idx - 1)) conj_mult(:,30*idx+1:90)];
```
16：49开始合成真实数据BVP750组

#### 4月4日

<img src="image/image4_4.png" style="zoom:50%;" />

算是看到些上升的趋势吧，但是准确率太低了。

#### 4月5日

写神经网络如下，不使用原代码中BVP，只使用r1的512*90矩阵
```
import os
import random
import numpy as np
from scipy.io import loadmat
import tensorflow as tf
from keras import layers
import re

data_folder = 'dataset/wifi/old2'
new_data_folder = 'dataset/wifi/output'
index = 1
class_labels = ['action1', 'action2', 'action3', 'action4', 'action5', 'action6']

data = []
labels = []

for file_name in os.listdir(data_folder):
    file_path = os.path.join(data_folder, file_name)
    if file_path.endswith('.mat'):
        gesture = int(file_name.split('-')[1])
        location = int(file_name.split('-')[2])
        orientation = int(file_name.split('-')[3])
        repetition = int(file_name.split('-')[4])
        # print(file_name)
        receiver = int(file_name.split('r')[2].split('.')[0])
        # print(receiver)

        if receiver != 1:  # 只接受r1
            continue

        mat_data = loadmat(file_path)

        data.append(mat_data['pred'].reshape(512, 90))
        labels.append(gesture-1)

for file_name in os.listdir(new_data_folder):
    file_path = os.path.join(new_data_folder, file_name)
    if file_path.endswith('.mat'):
        gesture = int(file_name.split('-')[1])
        location = int(file_name.split('-')[2])
        orientation = int(file_name.split('-')[3])
        repetition = int(file_name.split('-')[4])
        # print(file_name)
        receiver = int(file_name.split('r')[2].split('.')[0])
        # print(receiver)

        if receiver != 1 or repetition > index:  # 只接受r1
            continue

        mat_data = loadmat(file_path)

        data.append(mat_data['pred'].reshape(512, 90))
        labels.append(gesture-1)

data = np.array(data)
labels = np.array(labels)

print(data.shape)

data_size = len(data)
indices = np.arange(data_size)
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

train_ratio = 0.8  # 训练集比例
train_size = int(data_size * train_ratio)

train_data = data[:train_size]
train_labels = labels[:train_size]
test_data = data[train_size:]
test_labels = labels[train_size:]

train_data = np.expand_dims(train_data, axis=-1)
test_data = np.expand_dims(test_data, axis=-1)


num_classes = len(class_labels)
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

input_shape = train_data.shape[1:]
model = tf.keras.Sequential()
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 32
epochs = 10
model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, validation_split=0.1)

test_loss, test_accuracy = model.evaluate(test_data, test_labels)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)
```
<img src="image/Line_4_5.png" style="zoom:50%;" />

准确率从30%左右提高到40%以上


#### 4月7日

<img src="image/Line_4_7.png" style="zoom:50%;" />

继续增加合成数据直到1：1，到达一定水平后不再增加

#### 4月10日

对数据进行一些预处理，将complex64拆分成float数组传入模型


#### 4月24日

进行fmcw数据处理

```
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
    # print(adc_data.shape)
    filesize = len(adc_data)
    adc_data = adc_data.reshape(-1, 4)
    LVDS = np.zeros(shape=[filesize // 4, 2], dtype=np.complex64)
    LVDS[:, 0] = adc_data[:, 0] + 1.j * adc_data[:, 2]
    LVDS[:, 1] = adc_data[:, 1] + 1.j * adc_data[:, 3]
    LVDS = LVDS.reshape(-1)
    numChirps = filesize // (2 * numAdcSamples * numRX)
    # Chirp(一共收发了这些个Chirps), numTx*numSamples*numRx
    LVDS = LVDS[:np.prod(numChirps // numTx * numAdcSamples * numRX * numTx)].reshape(numChirps // numTx, numAdcSamples * numRX * numTx)
    # print(LVDS.shape)
    # 每个Chirp中分别包含了12个虚拟收发天线之间的数据 Chirps,12根虚拟天线接收的ADCSamples
    LVDS = LVDS.reshape(-1, numTx * numRX, numAdcSamples).swapaxes(1, 0)
    LVDS = LVDS.reshape(numTx * numRX, -1)
    return LVDS


def processSinglefile(fileName, numAdcSamples, numRX, numTx, Nfft1, Nfft2, numChirp, numFrame):
    adcDataList = []

    framecount = 0
    for ii in range(0, 1, 1):
        fileNamecur = fileName + "_Raw_" + str(ii) + ".bin"
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

            # print(adcDataoutshift.shape)
            file_name = folderProcessed + "/People" + str(peopleProcessing) + "_" + str(framecount) + ".mat"

            sio.savemat(file_name, {"label": peopleProcessing, "data": adcDataoutshift})

            adcDataList.append(adcDataoutshift)
            framecount += 1

```
只使用Empty的数据集，最终处理后的数据集（3\*4）\*128\*256\*（600\*22），3是发送天线数量，4是接收天线数量，128是每一帧的chirp数量，256是每一帧的采样率，共22个人，每一个人600帧数据。



#### 4月25日

数据处理，选择256\*128进行训练，参数调整，sample_rate=256，input_dim=128，cond_dim=1

#### 4月26日

数据量比较大，用时18h，训练4epoch，损失率1e-2

<img src="image/ssim_cdf.png" style="zoom:50%;" />
平均ssim为0.8182，CDF曲线如上图。


#### 4月27日

参考https://blog.csdn.net/Pin_BOY/article/details/116407502  ，尝试使用requirement.txt导入docker


#### 4月29日

ValueError: num_samples should be a positive integer value, but got num_samples=0

解决方法 https://blog.csdn.net/qq_38681990/article/details/119606840  ，windows和linux的pytorch的dataLoader些许差别。

RuntimeError: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx

解决方法https://www.cnblogs.com/chester-cs/p/14444247.html ，添加参数--gpus all

FileExistsError: [Errno 17] File exists: 'weights-3300.pt' -> './project/model/fmcw/b32-256-100s/weights.pt'

```
if os.name == 'nt':
    torch.save(self.state_dict(), link_name)
else:
    if os.path.islink(link_name):
        os.unlink(link_name)
    os.symlink(save_basename, link_name)
```
报错代码如上，可以看到针对windows和linux采用了不同的代码

删除build cache能提高镜像build速度， docker builder prune

创建镜像，docker build --no-cache -t wbfmcw .

运行镜像，docker run --rm --gpus all wbfmcw



