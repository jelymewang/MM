%针对属性OCC	 BC CM DEF FM HO LI LR MB NO PO SV TC ALL 画相应的precision和success曲线
close all;
clear all;
clc;
pixelThreshold=20;

attrMat='AttrMat_RGBT234.mat';%属性保存的mat

basePath='.\dataset\';
resultMatPath='ERRresults\';
attrDisplays = {'ALl'};%画某种属性的曲线  BC CM DEF FM HO LI LR MB NO PO SV TC ALL
%attrDisplays={'BC','CM','DEF','FM','HO','LI','LR','MB','NO','PO','SV','TC','ALL'};


algs={'SGT','DSST','SOWP','CSR','L1-PF','JSR','MEEM+RGBT','KCF+RGBT','CSR-DCF+RGBT','CFnet','CFnet+RGBT','SOWP+RGBT','ECO','C-COT','SRDCF','SAMF','CSR-DCF'};


attrs=load(attrMat);

colorStyle(:,:,1)=[1,0,0];colorStyle(:,:,2)=[0,0,1];colorStyle(:,:,3)=[0,1,0];colorStyle(:,:,4)=[0,1,1];colorStyle(:,:,5)=[1,0,1];
colorStyle(:,:,6)=[1,1,0];colorStyle(:,:,7)=[0.5,0,0];colorStyle(:,:,8)=[0,0.5,0];colorStyle(:,:,9)=[0,1,1];colorStyle(:,:,10)=[1,0,1];
colorStyle(:,:,11)=[1,0.5,0];colorStyle(:,:,12)=[0,0.5,1];colorStyle(:,:,13)=[0,1,0.5];colorStyle(:,:,14)=[0.5,1,1];colorStyle(:,:,15)=[1,0.5,1];
colorStyle(:,:,16)=[0.5,0,0];colorStyle(:,:,17)=[0,0,0.5];colorStyle(:,:,18)=[0,0.5,0];colorStyle(:,:,19)=[0,0.5,0.5];colorStyle(:,:,20)=[0.5,0,0.5];
lineStyle(:,:,1:8)='-';
lineStyle(:,:,9:16)=':';



sequencesAll=dir(basePath);
sequencesAll={sequencesAll.name};
sequencesAll=sequencesAll(3:end);

%得到那些包含AttrName的序列
for attr = 1:numel(attrDisplays)
    attrDisplay = attrDisplays{attr};
    sequences={};
    if strcmp(attrDisplay,'ALL')==1,
        sequences=sequencesAll;        
    else
        for seqIndex=1:length(sequencesAll),            
            idx=find(strcmp(attrs.SequencesName, sequencesAll{seqIndex})==1);%找到这个序列在mat中的下标
            switch attrDisplay,
                case 'BC';
                    if attrs.BC(idx)==1,
                        sequences{end+1}=sequencesAll{seqIndex};
                        attrName = 'Background Clutter';
                    end
                    
                case 'CM';
                    if attrs.CM(idx)==1,
                        sequences{end+1}=sequencesAll{seqIndex};
                        attrName = 'Camera Moving';
                    end
                case 'DEF';
                    if attrs.DEF(idx)==1,
                        sequences{end+1}=sequencesAll{seqIndex};
                        attrName = 'Deformation';
                    end
                case 'FM';
                    if attrs.FM(idx)==1,
                        sequences{end+1}=sequencesAll{seqIndex};
                        attrName = 'Fast Motion';
                    end
                case 'HO';
                    if attrs.HO(idx)==1,
                        sequences{end+1}=sequencesAll{seqIndex};
                        attrName = 'Heavy Occlusion';
                    end
                case 'LI';
                    if attrs.LI(idx)==1,
                        sequences{end+1}=sequencesAll{seqIndex};
                        attrName = 'Low Illumination';
                    end
                case 'LR';
                    if attrs.LR(idx)==1,
                        sequences{end+1}=sequencesAll{seqIndex};
                        attrName = 'Low Resolution';
                    end
                case 'MB';
                    if attrs.MB(idx)==1,
                        sequences{end+1}=sequencesAll{seqIndex};
                        attrName = 'Motion Blur';
                    end
                case 'NO';
                    if attrs.NO(idx)==1,
                        sequences{end+1}=sequencesAll{seqIndex};
                        attrName = 'No Occlusion';
                    end
                case 'PO';
                    if attrs.PO(idx)==1,
                        sequences{end+1}=sequencesAll{seqIndex};
                        attrName = 'Partial Occlusion';
                    end
                case 'SV';
                    if attrs.SC(idx)==1,
                        sequences{end+1}=sequencesAll{seqIndex};
                        attrName = 'Scale Variation';
                    end
                case 'TC';
                    if attrs.TC(idx)==1,
                        sequences{end+1}=sequencesAll{seqIndex};
                        attrName = 'Thermal Crossover';
                    end
            end
        end
    end
    
    
    disp([attrDisplay  ' seqNum:'  int2str(length(sequences))]);
    
    
    precisionX=0:1:50;
    successX=0:0.05:1;
    precisionY=zeros([size(precisionX,2) size(algs,2)]);
    successY=zeros([size(successX,2) size(algs,2)]);
    allFramesNum=0;
    
    for algIndex=1:size(algs,2),
        for seqIndex=1:size(sequences,2),
            results(seqIndex)=load([resultMatPath algs{algIndex} '/' algs{algIndex} '_' sequences{seqIndex} '.mat']);
            curr_frame_num=size(results(seqIndex).err,1);
            allFramesNum=allFramesNum+curr_frame_num;
            for j=1:size(precisionX,2),
                precisionY(j,algIndex)=precisionY(j,algIndex)+sum(results(seqIndex).errCenter<=precisionX(j))./curr_frame_num;
            end
            for j=1:size(successX,2),
                successY(j,algIndex)=successY(j,algIndex)+sum(results(seqIndex).err>successX(j))./curr_frame_num;
            end
        end
    end


    disp([attrDisplay 'FrameNum:' int2str(allFramesNum/length(algs))]);
    allFramesNum=allFramesNum/size(algs,2);
    seq_num = size(sequences,2);
    
    %.................................................................................
    %precision Plot
    %.................................................................................
    
    precisionY=precisionY./seq_num;
    precisionthr(1:size(algs,2))=precisionY(pixelThreshold+1,1:size(algs,2));
    [~,precisionIndex]=sort(precisionthr,'descend');
    
    h1=figure('Name',attrDisplay);
    for trackerIndex=1:size(algs,2),
        
        plot(precisionX,precisionY(:,precisionIndex(trackerIndex))','color',colorStyle(:,:,trackerIndex),'LineWidth',3,'LineStyle',lineStyle(:,:,trackerIndex));
        hold on
        precision=sprintf('%.3f', precisionY(pixelThreshold+1,precisionIndex(trackerIndex)));
        legendLabel{trackerIndex}=[algs{precisionIndex(trackerIndex)} '[' precision ']'];
    end
    
    if strcmp(attrDisplay,'ALL')~=1
        title([' Precision Plot - ',attrName]);
    else
        title('Precision Plot');
    end
    xlabel('Location error threshold','FontSize',20)
    ylabel('Maximum Precision Rate','FontSize',20)
    legend(legendLabel,'Location','northwest');  %southeast
    axis([0 50 0 1]);  % 设置坐标轴在指定的区间
    figName = ['.\figsResults\',attrDisplay,'_PR'];
    saveas(gcf,figName,'fig');
    
    
    %.................................................................................
    %success Plot
    %.................................................................................
    successY=successY./seq_num;
    successthr=mean(successY);
    [~,successIndex]=sort(successthr,'descend');
    
    h2=figure('Name',attrDisplay);
    for trackerIndex=1:size(algs,2),
        plot(successX,successY(:,successIndex(trackerIndex))','color',colorStyle(:,:,trackerIndex),'LineWidth',3,'LineStyle',lineStyle(:,:,trackerIndex));
        hold on
        
        area=sprintf('%.3f', successthr(successIndex(trackerIndex)));
        
        legendLabel1{trackerIndex}=[algs{successIndex(trackerIndex)} '[' area ']'];
    end
    
    if strcmp(attrDisplay,'ALL')~=1
        title(['Success Plot - ',attrName]);
    else
        title('Success Plot');
    end
    xlabel('overlap threshold','FontSize',20)
    ylabel('Maximum Success Rate','FontSize',20)
    legend(legendLabel1);
    axis([0 1.0 0 1.0]);  % 设置坐标轴在指定的区间
    
    figName = ['.\figsResults\',attrDisplay,'_SR'];
    saveas(gcf,figName,'fig');
end
    
    
