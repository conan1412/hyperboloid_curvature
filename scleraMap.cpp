// corneaMap.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <ctime>
#include <iostream>
#include <vector>
#include<windows.h>
#include "windows.h"
#include <ShlObj.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstring>        // for strcpy(), strcat()
#include <io.h>
#include <fstream>
#include <windows.h>
#include <ShlObj.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstring>
#include <io.h>
#include <fstream>
#include <direct.h>
#include "scleraMapCon.h"
#include <ctime>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

//extern "C" void test1(int num);

using namespace std;
using namespace cv;

template <class Type>
Type stringToNum(const string& str)
{
    istringstream iss(str);
    Type num;
    iss >> num;
    return num;
}

vector<string> split(const string& s, const string& seperator) {
    vector<string> result;
    typedef string::size_type string_size;
    string_size i = 0;

    while (i != s.size()) {
        //找到字符串中首个不等于分隔符的字母；
        int flag = 0;
        while (i != s.size() && flag == 0) {
            flag = 1;
            for (string_size x = 0; x < seperator.size(); ++x)
                if (s[i] == seperator[x]) {
                    ++i;
                    flag = 0;
                    break;
                }
        }

        //找到又一个分隔符，将两个分隔符之间的字符串取出；
        flag = 0;
        string_size j = i;
        while (j != s.size() && flag == 0) {
            for (string_size x = 0; x < seperator.size(); ++x)
                if (s[j] == seperator[x]) {
                    flag = 1;
                    break;
                }
            if (flag == 0)
                ++j;
        }
        if (i != j) {
            result.push_back(s.substr(i, j - i));
            i = j;
        }
    }
    return result;
}

void listFiles(const char* dir, const char* dir1, ScleraMap::Config config)
{
    const clock_t begin_time = clock();

    intptr_t handle;
    _finddata_t findData;

    handle = _findfirst(dir, &findData);    // 查找目录中的第一个文件
    if (handle == -1)
    {
        cout << "Failed to find first file!\n";
        return;
    }

    int number = 0;
    do
    {
        if (findData.attrib & _A_SUBDIR
            && strcmp(findData.name, ".") == 0
            && strcmp(findData.name, "..") == 0
            )
        {
            //cout << findData.name << endl;
        }// 是否是子目录并且不为"."或".."			
        else
        {
            //cout << findData.name << "\t" << findData.size << endl;
            if (number >= 0)
            {
                string path = string(dir1) + '\\' + findData.name;
                //cout << path << endl;
                vector<string> fn;
                glob(path, fn, false);
                if (fn.size() > 0)
                {
                    if (find(fn.begin(), fn.end(), string(path + "\\img_000.png")) != fn.end())
                    {
                        cout << path << endl;
                        //cout << "number:" << number << endl;
                        string filePath[12] = { path + "\\img_000.png",path + "\\img_001.png", path + "\\img_002.png",path + "\\img_003.png",path + "\\img_004.png",path + "\\img_005.png",path + "\\img_006.png",path + "\\img_007.png",path + "\\img_008.png",path + "\\img_009.png",path + "\\img_010.png",path + "\\img_011.png" };
                        string filePathOut[12] = { path + "\\sclera\\imgnew_000.png",path + "\\sclera\\imgnew_001.png", path + "\\sclera\\imgnew_002.png",path + "\\sclera\\imgnew_003.png",path + "\\sclera\\imgnew_004.png",path + "\\sclera\\imgnew_005.png",path + "\\sclera\\imgnew_006.png",path + "\\sclera\\imgnew_007.png",path + "\\sclera\\imgnew_008.png",path + "\\sclera\\imgnew_009.png",path + "\\sclera\\imgnew_010.png",path + "\\sclera\\imgnew_011.png" };
                        //string filePathOut[12] = { path + "\\result\\predict_000.png",path + "\\result\\predict_001.png", path + "\\result\\predict_002.png",path + "\\result\\predict_003.png",path + "\\result\\predict_004.png",path + "\\result\\predict_005.png",path + "\\result\\predict_006.png",path + "\\result\\predict_007.png",path + "\\result\\predict_008.png",path + "\\result\\predict_009.png",path + "\\result\\predict_010.png",path + "\\result\\predict_011.png" };
                        string name[12] = { "img_000.png",  "img_001.png",   "img_002.png",  "img_003.png",  "img_004.png",  "img_005.png", "img_006.png",  "img_007.png",  "img_008.png", "img_009.png",  "img_010.png",  "img_011.png" };
                        /*cout << path << endl;
                        cout << filePath[0] << endl;*/
                        int nPicNum = 12;

                        string command;
                        command = path + "\\sclera";
                        if (0 != _access(command.c_str(), 0))
                        {
                            // if this folder not exist, create a new one.
                            _mkdir(command.c_str());   // 返回 0 表示创建成功，-1 表示失败
                            //换成 ::_mkdir  ::_access 也行，不知道什么意思
                        }
                        
                        cout << "map constrution_______________" << endl;
                        
                        double R_sclera;
                        double R_cornea;
                        ScleraMap sclera;
                        sclera.setConfig(config);
                        sclera.mapCon(filePath, filePathOut, nPicNum, command, R_sclera,R_cornea);
                        sclera.r_sclera = R_sclera;
                        sclera.r_cornea = R_cornea;
                        cout << "R_cornea: " << R_cornea << endl;
                        //cout << "R_sclera: " << R_sclera << endl;
                        

                        float seconds = float(clock() - begin_time) / 1000;
                        cout << "module run time: " << seconds << endl;



                    }
                }





            }
            number = number + 1;




        }

    } while (_findnext(handle, &findData) == 0);    // 查找目录中的下一个文件

    cout << "Done!\n";
    _findclose(handle);    // 关闭搜索句柄
}



int main(int argc, char** argv)
{
    //test1(5000);

    char buffer[1024];
    if (_getcwd(buffer, sizeof(char) * 1024))
    {
        printf(buffer);
        cout << endl;
    }
    string path = string(buffer);

    vector<double> sample;
    ifstream ifstr_data(path + "//data.txt");
    double d;
    std::string str, s;

    s = ":";

    while (ifstr_data >> str)
    {
        std::vector<std::string> vec = split(str, s);
        sample.push_back(stringToNum<double>(vec[1]));//将数据压入堆栈。//
    }

    ifstr_data.close();



    ScleraMap::Config config;
    config.eye = (int)sample[0];
    config.mode = (int)sample[1];
    config.window = (int)sample[2];
    /*config.value1 = sample[3];
    config.value2 = sample[4];*/
    config.colorbarPos = (int)sample[3];
    config.width = sample[4];
    config.depth = sample[5];
    config.index = sample[6];
    config.nImgDeep = sample[7];
    config.nImgLength = sample[8];
    config.cutSize = sample[9];
    config.xCorr = sample[10];
    config.yCorr = sample[11];
    config.Rcornea = sample[12];
    config.Rup = sample[13];
    config.Rdw = sample[14];
    //config.if3D = sample[15];
    config.fineHeight_startidx = sample[16];  // 分割出的上界线往下平移多少像素
    config.fineHeight = sample[17];  // 分割区域的范围h
        



    
    bool ifCon = TRUE;
    while (ifCon)
    {



        char dir[200];
        char dir1[200];
        cout << "Enter a directory : ";
        cin.getline(dir, 200);
        strcpy_s(dir1, dir);

        //cin.getline(dir1, 200);
        strcat_s(dir, "\\*.*");
        cout << "---------------------calculating-------------------------------" << endl;

        listFiles(dir, dir1, config);

        char dir2[300];
        cout << "Enter 0 to exit or any button to continue: ";
        cin.getline(dir2, 200);
        string coin = string(dir2);
        if (coin == "0")
        {
            ifCon = FALSE;
        }
    }



    waitKey();

    return 0;
}


