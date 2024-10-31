    function targMap = targDataMap(),

    ;%***********************
    ;% Create Parameter Map *
    ;%***********************
    
        nTotData      = 0; %add to this count as we go
        nTotSects     = 1;
        sectIdxOffset = 0;

        ;%
        ;% Define dummy sections & preallocate arrays
        ;%
        dumSection.nData = -1;
        dumSection.data  = [];

        dumData.logicalSrcIdx = -1;
        dumData.dtTransOffset = -1;

        ;%
        ;% Init/prealloc paramMap
        ;%
        paramMap.nSections           = nTotSects;
        paramMap.sectIdxOffset       = sectIdxOffset;
            paramMap.sections(nTotSects) = dumSection; %prealloc
        paramMap.nTotData            = -1;

        ;%
        ;% Auto data (rtP)
        ;%
            section.nData     = 19;
            section.data(19)  = dumData; %prealloc

                    ;% rtP.CompareToConstant_const
                    section.data(1).logicalSrcIdx = 0;
                    section.data(1).dtTransOffset = 0;

                    ;% rtP.OutputFilter_A
                    section.data(2).logicalSrcIdx = 1;
                    section.data(2).dtTransOffset = 1;

                    ;% rtP.OutputFilter_C
                    section.data(3).logicalSrcIdx = 2;
                    section.data(3).dtTransOffset = 3;

                    ;% rtP.ZeroPole_A
                    section.data(4).logicalSrcIdx = 3;
                    section.data(4).dtTransOffset = 5;

                    ;% rtP.ZeroPole_B
                    section.data(5).logicalSrcIdx = 4;
                    section.data(5).dtTransOffset = 13;

                    ;% rtP.ZeroPole_C
                    section.data(6).logicalSrcIdx = 5;
                    section.data(6).dtTransOffset = 15;

                    ;% rtP.ZeroPole_D
                    section.data(7).logicalSrcIdx = 6;
                    section.data(7).dtTransOffset = 19;

                    ;% rtP.TransportDelay_Delay
                    section.data(8).logicalSrcIdx = 7;
                    section.data(8).dtTransOffset = 20;

                    ;% rtP.TransportDelay_InitOutput
                    section.data(9).logicalSrcIdx = 8;
                    section.data(9).dtTransOffset = 21;

                    ;% rtP.Bias_Bias
                    section.data(10).logicalSrcIdx = 9;
                    section.data(10).dtTransOffset = 22;

                    ;% rtP.Gain_Gain
                    section.data(11).logicalSrcIdx = 10;
                    section.data(11).dtTransOffset = 23;

                    ;% rtP.Gain_Gain_idyf3glywd
                    section.data(12).logicalSrcIdx = 11;
                    section.data(12).dtTransOffset = 24;

                    ;% rtP.SineWave_Amp
                    section.data(13).logicalSrcIdx = 12;
                    section.data(13).dtTransOffset = 25;

                    ;% rtP.SineWave_Bias
                    section.data(14).logicalSrcIdx = 13;
                    section.data(14).dtTransOffset = 26;

                    ;% rtP.SineWave_Freq
                    section.data(15).logicalSrcIdx = 14;
                    section.data(15).dtTransOffset = 27;

                    ;% rtP.SineWave_Phase
                    section.data(16).logicalSrcIdx = 15;
                    section.data(16).dtTransOffset = 28;

                    ;% rtP.Gain1_Gain
                    section.data(17).logicalSrcIdx = 16;
                    section.data(17).dtTransOffset = 29;

                    ;% rtP.Integrator_A
                    section.data(18).logicalSrcIdx = 17;
                    section.data(18).dtTransOffset = 30;

                    ;% rtP.Integrator_C
                    section.data(19).logicalSrcIdx = 18;
                    section.data(19).dtTransOffset = 34;

            nTotData = nTotData + section.nData;
            paramMap.sections(1) = section;
            clear section


            ;%
            ;% Non-auto Data (parameter)
            ;%


        ;%
        ;% Add final counts to struct.
        ;%
        paramMap.nTotData = nTotData;



    ;%**************************
    ;% Create Block Output Map *
    ;%**************************
    
        nTotData      = 0; %add to this count as we go
        nTotSects     = 1;
        sectIdxOffset = 0;

        ;%
        ;% Define dummy sections & preallocate arrays
        ;%
        dumSection.nData = -1;
        dumSection.data  = [];

        dumData.logicalSrcIdx = -1;
        dumData.dtTransOffset = -1;

        ;%
        ;% Init/prealloc sigMap
        ;%
        sigMap.nSections           = nTotSects;
        sigMap.sectIdxOffset       = sectIdxOffset;
            sigMap.sections(nTotSects) = dumSection; %prealloc
        sigMap.nTotData            = -1;

        ;%
        ;% Auto data (rtB)
        ;%
            section.nData     = 9;
            section.data(9)  = dumData; %prealloc

                    ;% rtB.opyihwpyem
                    section.data(1).logicalSrcIdx = 0;
                    section.data(1).dtTransOffset = 0;

                    ;% rtB.kdfvhapu1i
                    section.data(2).logicalSrcIdx = 1;
                    section.data(2).dtTransOffset = 1;

                    ;% rtB.aemqx1t5mh
                    section.data(3).logicalSrcIdx = 2;
                    section.data(3).dtTransOffset = 2;

                    ;% rtB.kzsfgqhpv1
                    section.data(4).logicalSrcIdx = 3;
                    section.data(4).dtTransOffset = 3;

                    ;% rtB.lwhumkefgn
                    section.data(5).logicalSrcIdx = 4;
                    section.data(5).dtTransOffset = 4;

                    ;% rtB.odv4zor5wc
                    section.data(6).logicalSrcIdx = 5;
                    section.data(6).dtTransOffset = 5;

                    ;% rtB.g1gyqhmuvp
                    section.data(7).logicalSrcIdx = 6;
                    section.data(7).dtTransOffset = 6;

                    ;% rtB.ozgjv5kcoq
                    section.data(8).logicalSrcIdx = 7;
                    section.data(8).dtTransOffset = 7;

                    ;% rtB.iotcjfxadz
                    section.data(9).logicalSrcIdx = 8;
                    section.data(9).dtTransOffset = 8;

            nTotData = nTotData + section.nData;
            sigMap.sections(1) = section;
            clear section


            ;%
            ;% Non-auto Data (signal)
            ;%


        ;%
        ;% Add final counts to struct.
        ;%
        sigMap.nTotData = nTotData;



    ;%*******************
    ;% Create DWork Map *
    ;%*******************
    
        nTotData      = 0; %add to this count as we go
        nTotSects     = 4;
        sectIdxOffset = 1;

        ;%
        ;% Define dummy sections & preallocate arrays
        ;%
        dumSection.nData = -1;
        dumSection.data  = [];

        dumData.logicalSrcIdx = -1;
        dumData.dtTransOffset = -1;

        ;%
        ;% Init/prealloc dworkMap
        ;%
        dworkMap.nSections           = nTotSects;
        dworkMap.sectIdxOffset       = sectIdxOffset;
            dworkMap.sections(nTotSects) = dumSection; %prealloc
        dworkMap.nTotData            = -1;

        ;%
        ;% Auto data (rtDW)
        ;%
            section.nData     = 1;
            section.data(1)  = dumData; %prealloc

                    ;% rtDW.nequhacir4.modelTStart
                    section.data(1).logicalSrcIdx = 0;
                    section.data(1).dtTransOffset = 0;

            nTotData = nTotData + section.nData;
            dworkMap.sections(1) = section;
            clear section

            section.nData     = 4;
            section.data(4)  = dumData; %prealloc

                    ;% rtDW.etzcdtgg4v.TUbufferPtrs
                    section.data(1).logicalSrcIdx = 1;
                    section.data(1).dtTransOffset = 0;

                    ;% rtDW.h4urlizny4.LoggedData
                    section.data(2).logicalSrcIdx = 2;
                    section.data(2).dtTransOffset = 2;

                    ;% rtDW.k3ahccqiwo.AQHandles
                    section.data(3).logicalSrcIdx = 3;
                    section.data(3).dtTransOffset = 5;

                    ;% rtDW.ozcp50hell.AQHandles
                    section.data(4).logicalSrcIdx = 4;
                    section.data(4).dtTransOffset = 6;

            nTotData = nTotData + section.nData;
            dworkMap.sections(2) = section;
            clear section

            section.nData     = 1;
            section.data(1)  = dumData; %prealloc

                    ;% rtDW.paeebttgho.Tail
                    section.data(1).logicalSrcIdx = 5;
                    section.data(1).dtTransOffset = 0;

            nTotData = nTotData + section.nData;
            dworkMap.sections(3) = section;
            clear section

            section.nData     = 1;
            section.data(1)  = dumData; %prealloc

                    ;% rtDW.e4drogkh1r
                    section.data(1).logicalSrcIdx = 6;
                    section.data(1).dtTransOffset = 0;

            nTotData = nTotData + section.nData;
            dworkMap.sections(4) = section;
            clear section


            ;%
            ;% Non-auto Data (dwork)
            ;%


        ;%
        ;% Add final counts to struct.
        ;%
        dworkMap.nTotData = nTotData;



    ;%
    ;% Add individual maps to base struct.
    ;%

    targMap.paramMap  = paramMap;
    targMap.signalMap = sigMap;
    targMap.dworkMap  = dworkMap;

    ;%
    ;% Add checksums to base struct.
    ;%


    targMap.checksum0 = 704014149;
    targMap.checksum1 = 2194897708;
    targMap.checksum2 = 2911416811;
    targMap.checksum3 = 3195465174;

