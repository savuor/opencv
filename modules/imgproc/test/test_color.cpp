/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

//to be removed
#include "opencv2/core.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/core/sse_utils.hpp"


using namespace cv;
using namespace std;

/////////////////////////// base test class for color transformations /////////////////////////

class CV_ColorCvtBaseTest : public cvtest::ArrayTest
{
public:
    CV_ColorCvtBaseTest( bool custom_inv_transform, bool allow_32f, bool allow_16u );

protected:
    int prepare_test_case( int test_case_idx );
    void prepare_to_validation( int /*test_case_idx*/ );
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void get_minmax_bounds( int i, int j, int type, Scalar& low, Scalar& high );

    // input --- fwd_transform -> ref_output[0]
    virtual void convert_forward( const Mat& src, Mat& dst );
    // ref_output[0] --- inv_transform ---> ref_output[1] (or input -- copy --> ref_output[1])
    virtual void convert_backward( const Mat& src, const Mat& dst, Mat& dst2 );

    // called from default implementation of convert_forward
    virtual void convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n );

    // called from default implementation of convert_backward
    virtual void convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n );

    const char* fwd_code_str;
    const char* inv_code_str;

    void run_func();
    bool allow_16u, allow_32f;
    int blue_idx;
    bool inplace;
    bool custom_inv_transform;
    int fwd_code, inv_code;
    bool test_cpp;
    int hue_range;
};


CV_ColorCvtBaseTest::CV_ColorCvtBaseTest( bool _custom_inv_transform, bool _allow_32f, bool _allow_16u )
{
    test_array[INPUT].push_back(NULL);
    test_array[OUTPUT].push_back(NULL);
    test_array[OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);
    allow_16u = _allow_16u;
    allow_32f = _allow_32f;
    custom_inv_transform = _custom_inv_transform;
    fwd_code = inv_code = -1;
    element_wise_relative_error = false;

    fwd_code_str = inv_code_str = 0;

    test_cpp = false;
    hue_range = 0;
    blue_idx = 0;
    inplace = false;
}


void CV_ColorCvtBaseTest::get_minmax_bounds( int i, int j, int type, Scalar& low, Scalar& high )
{
    cvtest::ArrayTest::get_minmax_bounds( i, j, type, low, high );
    if( i == INPUT )
    {
        int depth = CV_MAT_DEPTH(type);
        low = Scalar::all(0.);
        high = Scalar::all( depth == CV_8U ? 256 : depth == CV_16U ? 65536 : 1. );
    }
}


void CV_ColorCvtBaseTest::get_test_array_types_and_sizes( int test_case_idx,
                                                vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    int depth, cn;
    cvtest::ArrayTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    if( allow_16u && allow_32f )
    {
        depth = cvtest::randInt(rng) % 3;
        depth = depth == 0 ? CV_8U : depth == 1 ? CV_16U : CV_32F;
    }
    else if( allow_16u || allow_32f )
    {
        depth = cvtest::randInt(rng) % 2;
        depth = depth == 0 ? CV_8U : allow_16u ? CV_16U : CV_32F;
    }
    else
        depth = CV_8U;

    cn = (cvtest::randInt(rng) & 1) + 3;
    blue_idx = cvtest::randInt(rng) & 1 ? 2 : 0;

    types[INPUT][0] = CV_MAKETYPE(depth, cn);
    types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_MAKETYPE(depth, 3);
    if( test_array[OUTPUT].size() > 1 )
        types[OUTPUT][1] = types[REF_OUTPUT][1] = CV_MAKETYPE(depth, cn);

    inplace = cn == 3 && cvtest::randInt(rng) % 2 != 0;
    test_cpp = (cvtest::randInt(rng) & 256) == 0;
}


int CV_ColorCvtBaseTest::prepare_test_case( int test_case_idx )
{
    int code = cvtest::ArrayTest::prepare_test_case( test_case_idx );
    if( code > 0 && inplace )
        cvtest::copy( test_mat[INPUT][0], test_mat[OUTPUT][0] );
    return code;
}

void CV_ColorCvtBaseTest::run_func()
{
    CvArr* out0 = test_array[OUTPUT][0];
    cv::Mat _out0 = cv::cvarrToMat(out0), _out1 = cv::cvarrToMat(test_array[OUTPUT][1]);

    if(!test_cpp)
        cvCvtColor( inplace ? out0 : test_array[INPUT][0], out0, fwd_code );
    else
        cv::cvtColor( cv::cvarrToMat(inplace ? out0 : test_array[INPUT][0]), _out0, fwd_code, _out0.channels());

    if( inplace )
    {
        cvCopy( out0, test_array[OUTPUT][1] );
        out0 = test_array[OUTPUT][1];
    }
    if(!test_cpp)
        cvCvtColor( out0, test_array[OUTPUT][1], inv_code );
    else
        cv::cvtColor(cv::cvarrToMat(out0), _out1, inv_code, _out1.channels());
}


void CV_ColorCvtBaseTest::prepare_to_validation( int /*test_case_idx*/ )
{
    convert_forward( test_mat[INPUT][0], test_mat[REF_OUTPUT][0] );
    convert_backward( test_mat[INPUT][0], test_mat[REF_OUTPUT][0],
                      test_mat[REF_OUTPUT][1] );
    int depth = test_mat[REF_OUTPUT][0].depth();
    if( depth == CV_8U && hue_range )
    {
        for( int y = 0; y < test_mat[REF_OUTPUT][0].rows; y++ )
        {
            uchar* h0 = test_mat[REF_OUTPUT][0].ptr(y);
            uchar* h = test_mat[OUTPUT][0].ptr(y);

            for( int x = 0; x < test_mat[REF_OUTPUT][0].cols; x++, h0 += 3, h += 3 )
            {
                if( abs(*h - *h0) >= hue_range-1 && (*h <= 1 || *h0 <= 1) )
                    *h = *h0 = 0;
            }
        }
    }
}


void CV_ColorCvtBaseTest::convert_forward( const Mat& src, Mat& dst )
{
    const float c8u = 0.0039215686274509803f; // 1./255
    const float c16u = 1.5259021896696422e-005f; // 1./65535
    int depth = src.depth();
    int cn = src.channels(), dst_cn = dst.channels();
    int cols = src.cols, dst_cols_n = dst.cols*dst_cn;
    vector<float> _src_buf(src.cols*3);
    vector<float> _dst_buf(dst.cols*3);
    float* src_buf = &_src_buf[0];
    float* dst_buf = &_dst_buf[0];
    int i, j;

    assert( (cn == 3 || cn == 4) && (dst_cn == 3 || dst_cn == 1) );

    for( i = 0; i < src.rows; i++ )
    {
        switch( depth )
        {
        case CV_8U:
            {
                const uchar* src_row = src.ptr(i);
                uchar* dst_row = dst.ptr(i);

                for( j = 0; j < cols; j++ )
                {
                    src_buf[j*3] = src_row[j*cn + blue_idx]*c8u;
                    src_buf[j*3+1] = src_row[j*cn + 1]*c8u;
                    src_buf[j*3+2] = src_row[j*cn + (blue_idx^2)]*c8u;
                }

                convert_row_bgr2abc_32f_c3( src_buf, dst_buf, cols );

                for( j = 0; j < dst_cols_n; j++ )
                {
                    int t = cvRound( dst_buf[j] );
                    dst_row[j] = saturate_cast<uchar>(t);
                }
            }
            break;
        case CV_16U:
            {
                const ushort* src_row = src.ptr<ushort>(i);
                ushort* dst_row = dst.ptr<ushort>(i);

                for( j = 0; j < cols; j++ )
                {
                    src_buf[j*3] = src_row[j*cn + blue_idx]*c16u;
                    src_buf[j*3+1] = src_row[j*cn + 1]*c16u;
                    src_buf[j*3+2] = src_row[j*cn + (blue_idx^2)]*c16u;
                }

                convert_row_bgr2abc_32f_c3( src_buf, dst_buf, cols );

                for( j = 0; j < dst_cols_n; j++ )
                {
                    int t = cvRound( dst_buf[j] );
                    dst_row[j] = saturate_cast<ushort>(t);
                }
            }
            break;
        case CV_32F:
            {
                const float* src_row = src.ptr<float>(i);
                float* dst_row = dst.ptr<float>(i);

                for( j = 0; j < cols; j++ )
                {
                    src_buf[j*3] = src_row[j*cn + blue_idx];
                    src_buf[j*3+1] = src_row[j*cn + 1];
                    src_buf[j*3+2] = src_row[j*cn + (blue_idx^2)];
                }

                convert_row_bgr2abc_32f_c3( src_buf, dst_row, cols );
            }
            break;
        default:
            assert(0);
        }
    }
}


void CV_ColorCvtBaseTest::convert_row_bgr2abc_32f_c3( const float* /*src_row*/,
                                                      float* /*dst_row*/, int /*n*/ )
{
}


void CV_ColorCvtBaseTest::convert_row_abc2bgr_32f_c3( const float* /*src_row*/,
                                                      float* /*dst_row*/, int /*n*/ )
{
}


void CV_ColorCvtBaseTest::convert_backward( const Mat& src, const Mat& dst, Mat& dst2 )
{
    if( custom_inv_transform )
    {
        int depth = src.depth();
        int src_cn = dst.channels(), cn = dst2.channels();
        int cols_n = src.cols*src_cn, dst_cols = dst.cols;
        vector<float> _src_buf(src.cols*3);
        vector<float> _dst_buf(dst.cols*3);
        float* src_buf = &_src_buf[0];
        float* dst_buf = &_dst_buf[0];
        int i, j;

        assert( cn == 3 || cn == 4 );

        for( i = 0; i < src.rows; i++ )
        {
            switch( depth )
            {
            case CV_8U:
                {
                    const uchar* src_row = dst.ptr(i);
                    uchar* dst_row = dst2.ptr(i);

                    for( j = 0; j < cols_n; j++ )
                        src_buf[j] = src_row[j];

                    convert_row_abc2bgr_32f_c3( src_buf, dst_buf, dst_cols );

                    for( j = 0; j < dst_cols; j++ )
                    {
                        int b = cvRound( dst_buf[j*3]*255. );
                        int g = cvRound( dst_buf[j*3+1]*255. );
                        int r = cvRound( dst_buf[j*3+2]*255. );
                        dst_row[j*cn + blue_idx] = saturate_cast<uchar>(b);
                        dst_row[j*cn + 1] = saturate_cast<uchar>(g);
                        dst_row[j*cn + (blue_idx^2)] = saturate_cast<uchar>(r);
                        if( cn == 4 )
                            dst_row[j*cn + 3] = 255;
                    }
                }
                break;
            case CV_16U:
                {
                    const ushort* src_row = dst.ptr<ushort>(i);
                    ushort* dst_row = dst2.ptr<ushort>(i);

                    for( j = 0; j < cols_n; j++ )
                        src_buf[j] = src_row[j];

                    convert_row_abc2bgr_32f_c3( src_buf, dst_buf, dst_cols );

                    for( j = 0; j < dst_cols; j++ )
                    {
                        int b = cvRound( dst_buf[j*3]*65535. );
                        int g = cvRound( dst_buf[j*3+1]*65535. );
                        int r = cvRound( dst_buf[j*3+2]*65535. );
                        dst_row[j*cn + blue_idx] = saturate_cast<ushort>(b);
                        dst_row[j*cn + 1] = saturate_cast<ushort>(g);
                        dst_row[j*cn + (blue_idx^2)] = saturate_cast<ushort>(r);
                        if( cn == 4 )
                            dst_row[j*cn + 3] = 65535;
                    }
                }
                break;
            case CV_32F:
                {
                    const float* src_row = dst.ptr<float>(i);
                    float* dst_row = dst2.ptr<float>(i);

                    convert_row_abc2bgr_32f_c3( src_row, dst_buf, dst_cols );

                    for( j = 0; j < dst_cols; j++ )
                    {
                        float b = dst_buf[j*3];
                        float g = dst_buf[j*3+1];
                        float r = dst_buf[j*3+2];
                        dst_row[j*cn + blue_idx] = b;
                        dst_row[j*cn + 1] = g;
                        dst_row[j*cn + (blue_idx^2)] = r;
                        if( cn == 4 )
                            dst_row[j*cn + 3] = 1.f;
                    }
                }
                break;
            default:
                assert(0);
            }
        }
    }
    else
    {
        int i, j, k;
        int elem_size = (int)src.elemSize(), elem_size1 = (int)src.elemSize1();
        int width_n = src.cols*elem_size;

        for( i = 0; i < src.rows; i++ )
        {
            memcpy( dst2.ptr(i), src.ptr(i), width_n );
            if( src.channels() == 4 )
            {
                // clear the alpha channel
                uchar* ptr = dst2.ptr(i) + elem_size1*3;
                for( j = 0; j < width_n; j += elem_size )
                {
                    for( k = 0; k < elem_size1; k++ )
                        ptr[j + k] = 0;
                }
            }
        }
    }
}


#undef INIT_FWD_INV_CODES
#define INIT_FWD_INV_CODES( fwd, inv )          \
    fwd_code = CV_##fwd; inv_code = CV_##inv;   \
    fwd_code_str = #fwd; inv_code_str = #inv

//// rgb <=> gray
class CV_ColorGrayTest : public CV_ColorCvtBaseTest
{
public:
    CV_ColorGrayTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n );
    void convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n );
    double get_success_error_level( int test_case_idx, int i, int j );
};


CV_ColorGrayTest::CV_ColorGrayTest() : CV_ColorCvtBaseTest( true, true, true )
{
    INIT_FWD_INV_CODES( BGR2GRAY, GRAY2BGR );
}


void CV_ColorGrayTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    CV_ColorCvtBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    int cn = CV_MAT_CN(types[INPUT][0]);
    types[OUTPUT][0] = types[REF_OUTPUT][0] = types[INPUT][0] & CV_MAT_DEPTH_MASK;
    inplace = false;

    if( cn == 3 )
    {
        if( blue_idx == 0 )
            fwd_code = CV_BGR2GRAY, inv_code = CV_GRAY2BGR;
        else
            fwd_code = CV_RGB2GRAY, inv_code = CV_GRAY2RGB;
    }
    else
    {
        if( blue_idx == 0 )
            fwd_code = CV_BGRA2GRAY, inv_code = CV_GRAY2BGRA;
        else
            fwd_code = CV_RGBA2GRAY, inv_code = CV_GRAY2RGBA;
    }
}


double CV_ColorGrayTest::get_success_error_level( int /*test_case_idx*/, int i, int j )
{
    int depth = test_mat[i][j].depth();
    return depth == CV_8U ? 2 : depth == CV_16U ? 16 : 1e-5;
}


void CV_ColorGrayTest::convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = test_mat[INPUT][0].depth();
    double scale = depth == CV_8U ? 255 : depth == CV_16U ? 65535 : 1;
    double cr = 0.299*scale;
    double cg = 0.587*scale;
    double cb = 0.114*scale;
    int j;

    for( j = 0; j < n; j++ )
        dst_row[j] = (float)(src_row[j*3]*cb + src_row[j*3+1]*cg + src_row[j*3+2]*cr);
}


void CV_ColorGrayTest::convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n )
{
    int j, depth = test_mat[INPUT][0].depth();
    float scale = depth == CV_8U ? (1.f/255) : depth == CV_16U ? 1.f/65535 : 1.f;
    for( j = 0; j < n; j++ )
        dst_row[j*3] = dst_row[j*3+1] = dst_row[j*3+2] = src_row[j]*scale;
}


//// rgb <=> ycrcb
class CV_ColorYCrCbTest : public CV_ColorCvtBaseTest
{
public:
    CV_ColorYCrCbTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n );
    void convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n );
};


CV_ColorYCrCbTest::CV_ColorYCrCbTest() : CV_ColorCvtBaseTest( true, true, true )
{
    INIT_FWD_INV_CODES( BGR2YCrCb, YCrCb2BGR );
}


void CV_ColorYCrCbTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    CV_ColorCvtBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    if( blue_idx == 0 )
        fwd_code = CV_BGR2YCrCb, inv_code = CV_YCrCb2BGR;
    else
        fwd_code = CV_RGB2YCrCb, inv_code = CV_YCrCb2RGB;
}


double CV_ColorYCrCbTest::get_success_error_level( int /*test_case_idx*/, int i, int j )
{
    int depth = test_mat[i][j].depth();
    return depth == CV_8U ? 2 : depth == CV_16U ? 32 : 1e-3;
}


void CV_ColorYCrCbTest::convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = test_mat[INPUT][0].depth();
    double scale = depth == CV_8U ? 255 : depth == CV_16U ? 65535 : 1;
    double bias = depth == CV_8U ? 128 : depth == CV_16U ? 32768 : 0.5;

    double M[] = { 0.299, 0.587, 0.114,
                   0.49981,  -0.41853,  -0.08128,
                   -0.16864,  -0.33107,   0.49970 };
    int j;
    for( j = 0; j < 9; j++ )
        M[j] *= scale;

    for( j = 0; j < n*3; j += 3 )
    {
        double r = src_row[j+2];
        double g = src_row[j+1];
        double b = src_row[j];
        double y = M[0]*r + M[1]*g + M[2]*b;
        double cr = M[3]*r + M[4]*g + M[5]*b + bias;
        double cb = M[6]*r + M[7]*g + M[8]*b + bias;
        dst_row[j] = (float)y;
        dst_row[j+1] = (float)cr;
        dst_row[j+2] = (float)cb;
    }
}


void CV_ColorYCrCbTest::convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = test_mat[INPUT][0].depth();
    double bias = depth == CV_8U ? 128 : depth == CV_16U ? 32768 : 0.5;
    double scale = depth == CV_8U ? 1./255 : depth == CV_16U ? 1./65535 : 1;
    double M[] = { 1,   1.40252,  0,
                   1,  -0.71440,  -0.34434,
                   1,   0,   1.77305 };
    int j;
    for( j = 0; j < 9; j++ )
        M[j] *= scale;

    for( j = 0; j < n*3; j += 3 )
    {
        double y = src_row[j];
        double cr = src_row[j+1] - bias;
        double cb = src_row[j+2] - bias;
        double r = M[0]*y + M[1]*cr + M[2]*cb;
        double g = M[3]*y + M[4]*cr + M[5]*cb;
        double b = M[6]*y + M[7]*cr + M[8]*cb;
        dst_row[j] = (float)b;
        dst_row[j+1] = (float)g;
        dst_row[j+2] = (float)r;
    }
}


//// rgb <=> hsv
class CV_ColorHSVTest : public CV_ColorCvtBaseTest
{
public:
    CV_ColorHSVTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n );
    void convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n );
};


CV_ColorHSVTest::CV_ColorHSVTest() : CV_ColorCvtBaseTest( true, true, false )
{
    INIT_FWD_INV_CODES( BGR2HSV, HSV2BGR );
    hue_range = 180;
}


void CV_ColorHSVTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    CV_ColorCvtBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    RNG& rng = ts->get_rng();

    bool full_hrange = (rng.next() & 256) != 0;
    if( full_hrange )
    {
        if( blue_idx == 0 )
            fwd_code = CV_BGR2HSV_FULL, inv_code = CV_HSV2BGR_FULL;
        else
            fwd_code = CV_RGB2HSV_FULL, inv_code = CV_HSV2RGB_FULL;
        hue_range = 256;
    }
    else
    {
        if( blue_idx == 0 )
            fwd_code = CV_BGR2HSV, inv_code = CV_HSV2BGR;
        else
            fwd_code = CV_RGB2HSV, inv_code = CV_HSV2RGB;
        hue_range = 180;
    }
}


double CV_ColorHSVTest::get_success_error_level( int /*test_case_idx*/, int i, int j )
{
    int depth = test_mat[i][j].depth();
    return depth == CV_8U ? (j == 0 ? 4 : 16) : depth == CV_16U ? 32 : 1e-3;
}


void CV_ColorHSVTest::convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = test_mat[INPUT][0].depth();
    float h_scale = depth == CV_8U ? hue_range*30.f/180 : 60.f;
    float scale = depth == CV_8U ? 255.f : depth == CV_16U ? 65535.f : 1.f;
    int j;

    for( j = 0; j < n*3; j += 3 )
    {
        float r = src_row[j+2];
        float g = src_row[j+1];
        float b = src_row[j];
        float vmin = MIN(r,g);
        float v = MAX(r,g);
        float s, h, diff;
        vmin = MIN(vmin,b);
        v = MAX(v,b);
        diff = v - vmin;
        if( diff == 0 )
            s = h = 0;
        else
        {
            s = diff/(v + FLT_EPSILON);
            diff = 1.f/diff;

            h = r == v ? (g - b)*diff :
                g == v ? 2 + (b - r)*diff : 4 + (r - g)*diff;

            if( h < 0 )
                h += 6;
        }

        dst_row[j] = h*h_scale;
        dst_row[j+1] = s*scale;
        dst_row[j+2] = v*scale;
    }
}

// taken from http://www.cs.rit.edu/~ncs/color/t_convert.html
void CV_ColorHSVTest::convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = test_mat[INPUT][0].depth();
    float h_scale = depth == CV_8U ? 180/(hue_range*30.f) : 1.f/60;
    float scale = depth == CV_8U ? 1.f/255 : depth == CV_16U ? 1.f/65535 : 1;
    int j;

    for( j = 0; j < n*3; j += 3 )
    {
        float h = src_row[j]*h_scale;
        float s = src_row[j+1]*scale;
        float v = src_row[j+2]*scale;
        float r = v, g = v, b = v;

        if( h < 0 )
            h += 6;
        else if( h >= 6 )
            h -= 6;

        if( s != 0 )
        {
            int i = cvFloor(h);
            float f = h - i;
            float p = v*(1 - s);
            float q = v*(1 - s*f);
            float t = v*(1 - s*(1 - f));

            if( i == 0 )
                r = v, g = t, b = p;
            else if( i == 1 )
                r = q, g = v, b = p;
            else if( i == 2 )
                r = p, g = v, b = t;
            else if( i == 3 )
                r = p, g = q, b = v;
            else if( i == 4 )
                r = t, g = p, b = v;
            else
                r = v, g = p, b = q;
        }

        dst_row[j] = b;
        dst_row[j+1] = g;
        dst_row[j+2] = r;
    }
}


//// rgb <=> hls
class CV_ColorHLSTest : public CV_ColorCvtBaseTest
{
public:
    CV_ColorHLSTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n );
    void convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n );
};


CV_ColorHLSTest::CV_ColorHLSTest() : CV_ColorCvtBaseTest( true, true, false )
{
    INIT_FWD_INV_CODES( BGR2HLS, HLS2BGR );
    hue_range = 180;
}


void CV_ColorHLSTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    CV_ColorCvtBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    if( blue_idx == 0 )
        fwd_code = CV_BGR2HLS, inv_code = CV_HLS2BGR;
    else
        fwd_code = CV_RGB2HLS, inv_code = CV_HLS2RGB;
}


double CV_ColorHLSTest::get_success_error_level( int /*test_case_idx*/, int i, int j )
{
    int depth = test_mat[i][j].depth();
    return depth == CV_8U ? (j == 0 ? 4 : 16) : depth == CV_16U ? 32 : 1e-4;
}


void CV_ColorHLSTest::convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = test_mat[INPUT][0].depth();
    float h_scale = depth == CV_8U ? 30.f : 60.f;
    float scale = depth == CV_8U ? 255.f : depth == CV_16U ? 65535.f : 1.f;
    int j;

    for( j = 0; j < n*3; j += 3 )
    {
        float r = src_row[j+2];
        float g = src_row[j+1];
        float b = src_row[j];
        float vmin = MIN(r,g);
        float v = MAX(r,g);
        float s, h, l, diff;
        vmin = MIN(vmin,b);
        v = MAX(v,b);
        diff = v - vmin;

        if( diff == 0 )
            s = h = 0, l = v;
        else
        {
            l = (v + vmin)*0.5f;
            s = l <= 0.5f ? diff / (v + vmin) : diff / (2 - v - vmin);
            diff = 1.f/diff;

            h = r == v ? (g - b)*diff :
                g == v ? 2 + (b - r)*diff : 4 + (r - g)*diff;

            if( h < 0 )
                h += 6;
        }

        dst_row[j] = h*h_scale;
        dst_row[j+1] = l*scale;
        dst_row[j+2] = s*scale;
    }
}


void CV_ColorHLSTest::convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = test_mat[INPUT][0].depth();
    float h_scale = depth == CV_8U ? 1.f/30 : 1.f/60;
    float scale = depth == CV_8U ? 1.f/255 : depth == CV_16U ? 1.f/65535 : 1;
    int j;

    for( j = 0; j < n*3; j += 3 )
    {
        float h = src_row[j]*h_scale;
        float l = src_row[j+1]*scale;
        float s = src_row[j+2]*scale;
        float r = l, g = l, b = l;

        if( h < 0 )
            h += 6;
        else if( h >= 6 )
            h -= 6;

        if( s != 0 )
        {
            float m2 = l <= 0.5f ? l*(1.f + s) : l + s - l*s;
            float m1 = 2*l - m2;
            float h1 = h + 2;

            if( h1 >= 6 )
                h1 -= 6;
            if( h1 < 1 )
                r = m1 + (m2 - m1)*h1;
            else if( h1 < 3 )
                r = m2;
            else if( h1 < 4 )
                r = m1 + (m2 - m1)*(4 - h1);
            else
                r = m1;

            h1 = h;

            if( h1 < 1 )
                g = m1 + (m2 - m1)*h1;
            else if( h1 < 3 )
                g = m2;
            else if( h1 < 4 )
                g = m1 + (m2 - m1)*(4 - h1);
            else
                g = m1;

            h1 = h - 2;
            if( h1 < 0 )
                h1 += 6;

            if( h1 < 1 )
                b = m1 + (m2 - m1)*h1;
            else if( h1 < 3 )
                b = m2;
            else if( h1 < 4 )
                b = m1 + (m2 - m1)*(4 - h1);
            else
                b = m1;
        }

        dst_row[j] = b;
        dst_row[j+1] = g;
        dst_row[j+2] = r;
    }
}


static const double RGB2XYZ[] =
{
     0.412453, 0.357580, 0.180423,
     0.212671, 0.715160, 0.072169,
     0.019334, 0.119193, 0.950227
};


static const double XYZ2RGB[] =
{
    3.240479, -1.53715, -0.498535,
   -0.969256, 1.875991, 0.041556,
    0.055648, -0.204043, 1.057311
};

static const float Xn = 0.950456f;
static const float Zn = 1.088754f;


//// rgb <=> xyz
class CV_ColorXYZTest : public CV_ColorCvtBaseTest
{
public:
    CV_ColorXYZTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n );
    void convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n );
};


CV_ColorXYZTest::CV_ColorXYZTest() : CV_ColorCvtBaseTest( true, true, true )
{
    INIT_FWD_INV_CODES( BGR2XYZ, XYZ2BGR );
}


void CV_ColorXYZTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    CV_ColorCvtBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    if( blue_idx == 0 )
        fwd_code = CV_BGR2XYZ, inv_code = CV_XYZ2BGR;
    else
        fwd_code = CV_RGB2XYZ, inv_code = CV_XYZ2RGB;
}


double CV_ColorXYZTest::get_success_error_level( int /*test_case_idx*/, int i, int j )
{
    int depth = test_mat[i][j].depth();
    return depth == CV_8U ? (j == 0 ? 2 : 8) : depth == CV_16U ? (j == 0 ? 64 : 128) : 1e-1;
}


void CV_ColorXYZTest::convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = test_mat[INPUT][0].depth();
    double scale = depth == CV_8U ? 255 : depth == CV_16U ? 65535 : 1;

    double M[9];
    int j;
    for( j = 0; j < 9; j++ )
        M[j] = RGB2XYZ[j]*scale;

    for( j = 0; j < n*3; j += 3 )
    {
        double r = src_row[j+2];
        double g = src_row[j+1];
        double b = src_row[j];
        double x = M[0]*r + M[1]*g + M[2]*b;
        double y = M[3]*r + M[4]*g + M[5]*b;
        double z = M[6]*r + M[7]*g + M[8]*b;
        dst_row[j] = (float)x;
        dst_row[j+1] = (float)y;
        dst_row[j+2] = (float)z;
    }
}


void CV_ColorXYZTest::convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = test_mat[INPUT][0].depth();
    double scale = depth == CV_8U ? 1./255 : depth == CV_16U ? 1./65535 : 1;

    double M[9];
    int j;
    for( j = 0; j < 9; j++ )
        M[j] = XYZ2RGB[j]*scale;

    for( j = 0; j < n*3; j += 3 )
    {
        double x = src_row[j];
        double y = src_row[j+1];
        double z = src_row[j+2];
        double r = M[0]*x + M[1]*y + M[2]*z;
        double g = M[3]*x + M[4]*y + M[5]*z;
        double b = M[6]*x + M[7]*y + M[8]*z;
        dst_row[j] = (float)b;
        dst_row[j+1] = (float)g;
        dst_row[j+2] = (float)r;
    }
}


//// rgb <=> L*a*b*
class CV_ColorLabTest : public CV_ColorCvtBaseTest
{
public:
    CV_ColorLabTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n );
    void convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n );
};


CV_ColorLabTest::CV_ColorLabTest() : CV_ColorCvtBaseTest( true, true, false )
{
    INIT_FWD_INV_CODES( BGR2Lab, Lab2BGR );
}


void CV_ColorLabTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    CV_ColorCvtBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    if( blue_idx == 0 )
        fwd_code = CV_LBGR2Lab, inv_code = CV_Lab2LBGR;
    else
        fwd_code = CV_LRGB2Lab, inv_code = CV_Lab2LRGB;
}


double CV_ColorLabTest::get_success_error_level( int /*test_case_idx*/, int i, int j )
{
    int depth = test_mat[i][j].depth();
    return depth == CV_8U ? 16 : depth == CV_16U ? 32 : 1e-3;
}


static const double _1_3 = 0.333333333333;
const static float _1_3f = static_cast<float>(_1_3);


void CV_ColorLabTest::convert_row_bgr2abc_32f_c3(const float* src_row, float* dst_row, int n)
{
    int depth = test_mat[INPUT][0].depth();
    float Lscale = depth == CV_8U ? 255.f/100.f : depth == CV_16U ? 65535.f/100.f : 1.f;
    float ab_bias = depth == CV_8U ? 128.f : depth == CV_16U ? 32768.f : 0.f;
    float M[9];

    for (int j = 0; j < 9; j++ )
        M[j] = (float)RGB2XYZ[j];

    for (int x = 0; x < n*3; x += 3)
    {
        float R = src_row[x + 2];
        float G = src_row[x + 1];
        float B = src_row[x];

        float X = (R * M[0] + G * M[1] + B * M[2]) / Xn;
        float Y = R * M[3] + G * M[4] + B * M[5];
        float Z = (R * M[6] + G * M[7] + B * M[8]) / Zn;
        float fX = X > 0.008856f ? pow(X, _1_3f) :
            (7.787f * X + 16.f / 116.f);
        float fZ = Z > 0.008856f ? pow(Z, _1_3f):
            (7.787f * Z + 16.f / 116.f);

        float L = 0.0f, fY = 0.0f;
        if (Y > 0.008856f)
        {
            fY = pow(Y, _1_3f);
            L = 116.f * fY - 16.f;
        }
        else
        {
            fY = 7.787f * Y + 16.f / 116.f;
            L = 903.3f * Y;
        }

        float a = 500.f * (fX - fY);
        float b = 200.f * (fY - fZ);

        dst_row[x] = L * Lscale;
        dst_row[x + 1] = a + ab_bias;
        dst_row[x + 2] = b + ab_bias;
    }
}

void CV_ColorLabTest::convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = test_mat[INPUT][0].depth();
    float Lscale = depth == CV_8U ? 100.f/255.f : depth == CV_16U ? 100.f/65535.f : 1.f;
    float ab_bias = depth == CV_8U ? 128.f : depth == CV_16U ? 32768.f : 0.f;
    float M[9];

    for(int j = 0; j < 9; j++ )
        M[j] = (float)XYZ2RGB[j];

    static const float lthresh = 903.3f * 0.008856f;
    static const float thresh = 7.787f * 0.008856f + 16.0f / 116.0f;
    for (int x = 0, end = n * 3; x < end; x += 3)
    {
        float L = src_row[x] * Lscale;
        float a = src_row[x + 1] - ab_bias;
        float b = src_row[x + 2] - ab_bias;

        float FY = 0.0f, Y = 0.0f;
        if (L <= lthresh)
        {
            Y = L / 903.3f;
            FY = 7.787f * Y + 16.0f / 116.0f;
        }
        else
        {
            FY = (L + 16.0f) / 116.0f;
            Y = FY * FY * FY;
        }

        float FX = a / 500.0f + FY;
        float FZ = FY - b / 200.0f;

        float FXZ[] = { FX, FZ };
        for (int k = 0; k < 2; ++k)
        {
            if (FXZ[k] <= thresh)
                FXZ[k] = (FXZ[k] - 16.0f / 116.0f) / 7.787f;
            else
                FXZ[k] = FXZ[k] * FXZ[k] * FXZ[k];
        }
        float X = FXZ[0] * Xn;
        float Z = FXZ[1] * Zn;

        float R = M[0] * X + M[1] * Y + M[2] * Z;
        float G = M[3] * X + M[4] * Y + M[5] * Z;
        float B = M[6] * X + M[7] * Y + M[8] * Z;

        dst_row[x] = B;
        dst_row[x + 1] = G;
        dst_row[x + 2] = R;
    }
}


//// rgb <=> L*u*v*
class CV_ColorLuvTest : public CV_ColorCvtBaseTest
{
public:
    CV_ColorLuvTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n );
    void convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n );
};


CV_ColorLuvTest::CV_ColorLuvTest() : CV_ColorCvtBaseTest( true, true, false )
{
    INIT_FWD_INV_CODES( BGR2Luv, Luv2BGR );
}


void CV_ColorLuvTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    CV_ColorCvtBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    if( blue_idx == 0 )
        fwd_code = CV_LBGR2Luv, inv_code = CV_Luv2LBGR;
    else
        fwd_code = CV_LRGB2Luv, inv_code = CV_Luv2LRGB;
}


double CV_ColorLuvTest::get_success_error_level( int /*test_case_idx*/, int i, int j )
{
    int depth = test_mat[i][j].depth();
    return depth == CV_8U ? 48 : depth == CV_16U ? 32 : 5e-2;
}


void CV_ColorLuvTest::convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = test_mat[INPUT][0].depth();
    float Lscale = depth == CV_8U ? 255.f/100.f : depth == CV_16U ? 65535.f/100.f : 1.f;
    int j;

    float M[9];
    float un = 4.f*Xn/(Xn + 15.f*1.f + 3*Zn);
    float vn = 9.f*1.f/(Xn + 15.f*1.f + 3*Zn);
    float u_scale = 1.f, u_bias = 0.f;
    float v_scale = 1.f, v_bias = 0.f;

    for( j = 0; j < 9; j++ )
        M[j] = (float)RGB2XYZ[j];

    if( depth == CV_8U )
    {
        u_scale = 0.720338983f;
        u_bias = 96.5254237f;
        v_scale = 0.973282442f;
        v_bias = 136.2595419f;
    }

    for( j = 0; j < n*3; j += 3 )
    {
        float r = src_row[j+2];
        float g = src_row[j+1];
        float b = src_row[j];

        float X = r*M[0] + g*M[1] + b*M[2];
        float Y = r*M[3] + g*M[4] + b*M[5];
        float Z = r*M[6] + g*M[7] + b*M[8];
        float d = X + 15*Y + 3*Z, L, u, v;

        if( d == 0 )
            L = u = v = 0;
        else
        {
            if( Y > 0.008856f )
                L = (float)(116.*pow((double)Y,_1_3) - 16.);
            else
                L = 903.3f * Y;

            d = 1.f/d;
            u = 13*L*(4*X*d - un);
            v = 13*L*(9*Y*d - vn);
        }
        dst_row[j] = L*Lscale;
        dst_row[j+1] = u*u_scale + u_bias;
        dst_row[j+2] = v*v_scale + v_bias;
    }
}


void CV_ColorLuvTest::convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = test_mat[INPUT][0].depth();
    float Lscale = depth == CV_8U ? 100.f/255.f : depth == CV_16U ? 100.f/65535.f : 1.f;
    int j;
    float M[9];
    float un = 4.f*Xn/(Xn + 15.f*1.f + 3*Zn);
    float vn = 9.f*1.f/(Xn + 15.f*1.f + 3*Zn);
    float u_scale = 1.f, u_bias = 0.f;
    float v_scale = 1.f, v_bias = 0.f;

    for( j = 0; j < 9; j++ )
        M[j] = (float)XYZ2RGB[j];

    if( depth == CV_8U )
    {
        u_scale = 1.f/0.720338983f;
        u_bias = 96.5254237f;
        v_scale = 1.f/0.973282442f;
        v_bias = 136.2595419f;
    }

    for( j = 0; j < n*3; j += 3 )
    {
        float L = src_row[j]*Lscale;
        float u = (src_row[j+1] - u_bias)*u_scale;
        float v = (src_row[j+2] - v_bias)*v_scale;
        float X, Y, Z;

        if( L >= 8 )
        {
            Y = (L + 16.f)*(1.f/116.f);
            Y = Y*Y*Y;
        }
        else
        {
            Y = L * (1.f/903.3f);
            if( L == 0 )
                L = 0.001f;
        }

        u = u/(13*L) + un;
        v = v/(13*L) + vn;

        X = -9*Y*u/((u - 4)*v - u*v);
        Z = (9*Y - 15*v*Y - v*X)/(3*v);

        float r = M[0]*X + M[1]*Y + M[2]*Z;
        float g = M[3]*X + M[4]*Y + M[5]*Z;
        float b = M[6]*X + M[7]*Y + M[8]*Z;

        dst_row[j] = b;
        dst_row[j+1] = g;
        dst_row[j+2] = r;
    }
}


//// rgb <=> another rgb
class CV_ColorRGBTest : public CV_ColorCvtBaseTest
{
public:
    CV_ColorRGBTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void convert_forward( const Mat& src, Mat& dst );
    void convert_backward( const Mat& src, const Mat& dst, Mat& dst2 );
    int dst_bits;
};


CV_ColorRGBTest::CV_ColorRGBTest() : CV_ColorCvtBaseTest( true, true, true )
{
    dst_bits = 0;
}


void CV_ColorRGBTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    CV_ColorCvtBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    int cn = CV_MAT_CN(types[INPUT][0]);

    dst_bits = 24;

    if( cvtest::randInt(rng) % 3 == 0 )
    {
        types[INPUT][0] = types[OUTPUT][1] = types[REF_OUTPUT][1] = CV_MAKETYPE(CV_8U,cn);
        types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_MAKETYPE(CV_8U,2);
        if( cvtest::randInt(rng) & 1 )
        {
            if( blue_idx == 0 )
                fwd_code = CV_BGR2BGR565, inv_code = CV_BGR5652BGR;
            else
                fwd_code = CV_RGB2BGR565, inv_code = CV_BGR5652RGB;
            dst_bits = 16;
        }
        else
        {
            if( blue_idx == 0 )
                fwd_code = CV_BGR2BGR555, inv_code = CV_BGR5552BGR;
            else
                fwd_code = CV_RGB2BGR555, inv_code = CV_BGR5552RGB;
            dst_bits = 15;
        }
    }
    else
    {
        if( cn == 3 )
        {
            fwd_code = CV_RGB2BGR, inv_code = CV_BGR2RGB;
            blue_idx = 2;
        }
        else if( blue_idx == 0 )
            fwd_code = CV_BGRA2BGR, inv_code = CV_BGR2BGRA;
        else
            fwd_code = CV_RGBA2BGR, inv_code = CV_BGR2RGBA;
    }

    if( CV_MAT_CN(types[INPUT][0]) != CV_MAT_CN(types[OUTPUT][0]) )
        inplace = false;
}


double CV_ColorRGBTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return 0;
}


void CV_ColorRGBTest::convert_forward( const Mat& src, Mat& dst )
{
    int depth = src.depth(), cn = src.channels();
/*#if defined _DEBUG || defined DEBUG
    int dst_cn = CV_MAT_CN(dst->type);
#endif*/
    int i, j, cols = src.cols;
    int g_rshift = dst_bits == 16 ? 2 : 3;
    int r_lshift = dst_bits == 16 ? 11 : 10;

    //assert( (cn == 3 || cn == 4) && (dst_cn == 3 || (dst_cn == 2 && depth == CV_8U)) );

    for( i = 0; i < src.rows; i++ )
    {
        switch( depth )
        {
        case CV_8U:
            {
                const uchar* src_row = src.ptr(i);
                uchar* dst_row = dst.ptr(i);

                if( dst_bits == 24 )
                {
                    for( j = 0; j < cols; j++ )
                    {
                        uchar b = src_row[j*cn + blue_idx];
                        uchar g = src_row[j*cn + 1];
                        uchar r = src_row[j*cn + (blue_idx^2)];
                        dst_row[j*3] = b;
                        dst_row[j*3+1] = g;
                        dst_row[j*3+2] = r;
                    }
                }
                else
                {
                    for( j = 0; j < cols; j++ )
                    {
                        int b = src_row[j*cn + blue_idx] >> 3;
                        int g = src_row[j*cn + 1] >> g_rshift;
                        int r = src_row[j*cn + (blue_idx^2)] >> 3;
                        ((ushort*)dst_row)[j] = (ushort)(b | (g << 5) | (r << r_lshift));
                        if( cn == 4 && src_row[j*4+3] )
                            ((ushort*)dst_row)[j] |= 1 << (r_lshift+5);
                    }
                }
            }
            break;
        case CV_16U:
            {
                const ushort* src_row = src.ptr<ushort>(i);
                ushort* dst_row = dst.ptr<ushort>(i);

                for( j = 0; j < cols; j++ )
                {
                    ushort b = src_row[j*cn + blue_idx];
                    ushort g = src_row[j*cn + 1];
                    ushort r = src_row[j*cn + (blue_idx^2)];
                    dst_row[j*3] = b;
                    dst_row[j*3+1] = g;
                    dst_row[j*3+2] = r;
                }
            }
            break;
        case CV_32F:
            {
                const float* src_row = src.ptr<float>(i);
                float* dst_row = dst.ptr<float>(i);

                for( j = 0; j < cols; j++ )
                {
                    float b = src_row[j*cn + blue_idx];
                    float g = src_row[j*cn + 1];
                    float r = src_row[j*cn + (blue_idx^2)];
                    dst_row[j*3] = b;
                    dst_row[j*3+1] = g;
                    dst_row[j*3+2] = r;
                }
            }
            break;
        default:
            assert(0);
        }
    }
}


void CV_ColorRGBTest::convert_backward( const Mat& /*src*/, const Mat& src, Mat& dst )
{
    int depth = src.depth(), cn = dst.channels();
/*#if defined _DEBUG || defined DEBUG
    int src_cn = CV_MAT_CN(src->type);
#endif*/
    int i, j, cols = src.cols;
    int g_lshift = dst_bits == 16 ? 2 : 3;
    int r_rshift = dst_bits == 16 ? 11 : 10;

    //assert( (cn == 3 || cn == 4) && (src_cn == 3 || (src_cn == 2 && depth == CV_8U)) );

    for( i = 0; i < src.rows; i++ )
    {
        switch( depth )
        {
        case CV_8U:
            {
                const uchar* src_row = src.ptr(i);
                uchar* dst_row = dst.ptr(i);

                if( dst_bits == 24 )
                {
                    for( j = 0; j < cols; j++ )
                    {
                        uchar b = src_row[j*3];
                        uchar g = src_row[j*3 + 1];
                        uchar r = src_row[j*3 + 2];

                        dst_row[j*cn + blue_idx] = b;
                        dst_row[j*cn + 1] = g;
                        dst_row[j*cn + (blue_idx^2)] = r;

                        if( cn == 4 )
                            dst_row[j*cn + 3] = 255;
                    }
                }
                else
                {
                    for( j = 0; j < cols; j++ )
                    {
                        ushort val = ((ushort*)src_row)[j];
                        uchar b = (uchar)(val << 3);
                        uchar g = (uchar)((val >> 5) << g_lshift);
                        uchar r = (uchar)((val >> r_rshift) << 3);

                        dst_row[j*cn + blue_idx] = b;
                        dst_row[j*cn + 1] = g;
                        dst_row[j*cn + (blue_idx^2)] = r;

                        if( cn == 4 )
                        {
                            uchar alpha = r_rshift == 11 || (val & 0x8000) != 0 ? 255 : 0;
                            dst_row[j*cn + 3] = alpha;
                        }
                    }
                }
            }
            break;
        case CV_16U:
            {
                const ushort* src_row = src.ptr<ushort>(i);
                ushort* dst_row = dst.ptr<ushort>(i);

                for( j = 0; j < cols; j++ )
                {
                    ushort b = src_row[j*3];
                    ushort g = src_row[j*3 + 1];
                    ushort r = src_row[j*3 + 2];

                    dst_row[j*cn + blue_idx] = b;
                    dst_row[j*cn + 1] = g;
                    dst_row[j*cn + (blue_idx^2)] = r;

                    if( cn == 4 )
                        dst_row[j*cn + 3] = 65535;
                }
            }
            break;
        case CV_32F:
            {
                const float* src_row = src.ptr<float>(i);
                float* dst_row = dst.ptr<float>(i);

                for( j = 0; j < cols; j++ )
                {
                    float b = src_row[j*3];
                    float g = src_row[j*3 + 1];
                    float r = src_row[j*3 + 2];

                    dst_row[j*cn + blue_idx] = b;
                    dst_row[j*cn + 1] = g;
                    dst_row[j*cn + (blue_idx^2)] = r;

                    if( cn == 4 )
                        dst_row[j*cn + 3] = 1.f;
                }
            }
            break;
        default:
            assert(0);
        }
    }
}


//// rgb <=> bayer

class CV_ColorBayerTest : public CV_ColorCvtBaseTest
{
public:
    CV_ColorBayerTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void run_func();
    void prepare_to_validation( int test_case_idx );
};


CV_ColorBayerTest::CV_ColorBayerTest() : CV_ColorCvtBaseTest( false, false, true )
{
    test_array[OUTPUT].pop_back();
    test_array[REF_OUTPUT].pop_back();

    fwd_code_str = "BayerBG2BGR";
    inv_code_str = "";
    fwd_code = CV_BayerBG2BGR;
    inv_code = -1;
}


void CV_ColorBayerTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    CV_ColorCvtBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    types[INPUT][0] = CV_MAT_DEPTH(types[INPUT][0]);
    types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_MAKETYPE(CV_MAT_DEPTH(types[INPUT][0]), 3);
    inplace = false;

    fwd_code = cvtest::randInt(rng)%4 + CV_BayerBG2BGR;
}


double CV_ColorBayerTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return 1;
}


void CV_ColorBayerTest::run_func()
{
    if(!test_cpp)
        cvCvtColor( test_array[INPUT][0], test_array[OUTPUT][0], fwd_code );
    else
    {
        cv::Mat _out = cv::cvarrToMat(test_array[OUTPUT][0]);
        cv::cvtColor(cv::cvarrToMat(test_array[INPUT][0]), _out, fwd_code, _out.channels());
    }
}


template<typename T>
static void bayer2BGR_(const Mat& src, Mat& dst, int code)
{
    int i, j, cols = src.cols - 2;
    int bi = 0;
    int step = (int)(src.step/sizeof(T));

    if( code == CV_BayerRG2BGR || code == CV_BayerGR2BGR )
        bi ^= 2;

    for( i = 1; i < src.rows - 1; i++ )
    {
        const T* ptr = src.ptr<T>(i) + 1;
        T* dst_row = dst.ptr<T>(i) + 3;
        int save_code = code;
        if( cols <= 0 )
        {
            dst_row[-3] = dst_row[-2] = dst_row[-1] = 0;
            dst_row[cols*3] = dst_row[cols*3+1] = dst_row[cols*3+2] = 0;
            continue;
        }

        for( j = 0; j < cols; j++ )
        {
            int b, g, r;
            if( !(code & 1) )
            {
                b = ptr[j];
                g = (ptr[j-1] + ptr[j+1] + ptr[j-step] + ptr[j+step])>>2;
                r = (ptr[j-step-1] + ptr[j-step+1] + ptr[j+step-1] + ptr[j+step+1]) >> 2;
            }
            else
            {
                b = (ptr[j-1] + ptr[j+1]) >> 1;
                g = ptr[j];
                r = (ptr[j-step] + ptr[j+step]) >> 1;
            }
            code ^= 1;
            dst_row[j*3 + bi] = (T)b;
            dst_row[j*3 + 1] = (T)g;
            dst_row[j*3 + (bi^2)] = (T)r;
        }

        dst_row[-3] = dst_row[0];
        dst_row[-2] = dst_row[1];
        dst_row[-1] = dst_row[2];
        dst_row[cols*3] = dst_row[cols*3-3];
        dst_row[cols*3+1] = dst_row[cols*3-2];
        dst_row[cols*3+2] = dst_row[cols*3-1];

        code = save_code ^ 1;
        bi ^= 2;
    }

    if( src.rows <= 2 )
    {
        memset( dst.ptr(), 0, (cols+2)*3*sizeof(T) );
        memset( dst.ptr(dst.rows-1), 0, (cols+2)*3*sizeof(T) );
    }
    else
    {
        T* top_row = dst.ptr<T>();
        T* bottom_row = dst.ptr<T>(dst.rows-1);
        int dstep = (int)(dst.step/sizeof(T));

        for( j = 0; j < (cols+2)*3; j++ )
        {
            top_row[j] = top_row[j + dstep];
            bottom_row[j] = bottom_row[j - dstep];
        }
    }
}


void CV_ColorBayerTest::prepare_to_validation( int /*test_case_idx*/ )
{
    const Mat& src = test_mat[INPUT][0];
    Mat& dst = test_mat[REF_OUTPUT][0];
    int depth = src.depth();
    if( depth == CV_8U )
        bayer2BGR_<uchar>(src, dst, fwd_code);
    else if( depth == CV_16U )
        bayer2BGR_<ushort>(src, dst, fwd_code);
    else
        CV_Error(CV_StsUnsupportedFormat, "");
}


/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Imgproc_ColorGray, accuracy) { CV_ColorGrayTest test; test.safe_run(); }
TEST(Imgproc_ColorYCrCb, accuracy) { CV_ColorYCrCbTest test; test.safe_run(); }
TEST(Imgproc_ColorHSV, accuracy) { CV_ColorHSVTest test; test.safe_run(); }
TEST(Imgproc_ColorHLS, accuracy) { CV_ColorHLSTest test; test.safe_run(); }
TEST(Imgproc_ColorXYZ, accuracy) { CV_ColorXYZTest test; test.safe_run(); }
TEST(Imgproc_ColorLab, accuracy) { CV_ColorLabTest test; test.safe_run(); }
TEST(Imgproc_ColorLuv, accuracy) { CV_ColorLuvTest test; test.safe_run(); }
TEST(Imgproc_ColorRGB, accuracy) { CV_ColorRGBTest test; test.safe_run(); }
TEST(Imgproc_ColorBayer, accuracy) { CV_ColorBayerTest test; test.safe_run(); }

TEST(Imgproc_ColorBayer, regression)
{
    cvtest::TS* ts = cvtest::TS::ptr();

    Mat given = imread(string(ts->get_data_path()) + "/cvtcolor/bayer_input.png", IMREAD_GRAYSCALE);
    Mat gold = imread(string(ts->get_data_path()) + "/cvtcolor/bayer_gold.png", IMREAD_UNCHANGED);
    Mat result;

    CV_Assert( !given.empty() && !gold.empty() );

    cvtColor(given, result, CV_BayerBG2GRAY);

    EXPECT_EQ(gold.type(), result.type());
    EXPECT_EQ(gold.cols, result.cols);
    EXPECT_EQ(gold.rows, result.rows);

    Mat diff;
    absdiff(gold, result, diff);

    EXPECT_EQ(0, countNonZero(diff.reshape(1) > 1));
}

TEST(Imgproc_ColorBayerVNG, regression)
{
    cvtest::TS* ts = cvtest::TS::ptr();

    Mat given = imread(string(ts->get_data_path()) + "/cvtcolor/bayer_input.png", IMREAD_GRAYSCALE);
    string goldfname = string(ts->get_data_path()) + "/cvtcolor/bayerVNG_gold.png";
    Mat gold = imread(goldfname, IMREAD_UNCHANGED);
    Mat result;

    CV_Assert( !given.empty() );

    cvtColor(given, result, CV_BayerBG2BGR_VNG, 3);

    if (gold.empty())
        imwrite(goldfname, result);
    else
    {
        EXPECT_EQ(gold.type(), result.type());
        EXPECT_EQ(gold.cols, result.cols);
        EXPECT_EQ(gold.rows, result.rows);

        Mat diff;
        absdiff(gold, result, diff);

        EXPECT_EQ(0, countNonZero(diff.reshape(1) > 1));
    }
}

// creating Bayer pattern
template <typename T, int depth>
static void calculateBayerPattern(const Mat& src, Mat& bayer, const char* pattern)
{
    Size ssize = src.size();
    const int scn = 1;
    bayer.create(ssize, CV_MAKETYPE(depth, scn));

    if (!strcmp(pattern, "bg"))
    {
        for (int y = 0; y < ssize.height; ++y)
            for (int x = 0; x < ssize.width; ++x)
            {
                if ((x + y) % 2)
                    bayer.at<T>(y, x) = static_cast<T>(src.at<Vec3b>(y, x)[1]);
                else if (x % 2)
                    bayer.at<T>(y, x) = static_cast<T>(src.at<Vec3b>(y, x)[0]);
                else
                    bayer.at<T>(y, x) = static_cast<T>(src.at<Vec3b>(y, x)[2]);
            }
    }
    else if (!strcmp(pattern, "gb"))
    {
        for (int y = 0; y < ssize.height; ++y)
            for (int x = 0; x < ssize.width; ++x)
            {
                if ((x + y) % 2 == 0)
                    bayer.at<T>(y, x) = static_cast<T>(src.at<Vec3b>(y, x)[1]);
                else if (x % 2 == 0)
                    bayer.at<T>(y, x) = static_cast<T>(src.at<Vec3b>(y, x)[0]);
                else
                    bayer.at<T>(y, x) = static_cast<T>(src.at<Vec3b>(y, x)[2]);
            }
    }
    else if (!strcmp(pattern, "rg"))
    {
        for (int y = 0; y < ssize.height; ++y)
            for (int x = 0; x < ssize.width; ++x)
            {
                if ((x + y) % 2)
                    bayer.at<T>(y, x) = static_cast<T>(src.at<Vec3b>(y, x)[1]);
                else if (x % 2 == 0)
                    bayer.at<T>(y, x) = static_cast<T>(src.at<Vec3b>(y, x)[0]);
                else
                    bayer.at<T>(y, x) = static_cast<T>(src.at<Vec3b>(y, x)[2]);
            }
    }
    else
    {
        for (int y = 0; y < ssize.height; ++y)
            for (int x = 0; x < ssize.width; ++x)
            {
                if ((x + y) % 2 == 0)
                    bayer.at<T>(y, x) = static_cast<T>(src.at<Vec3b>(y, x)[1]);
                else if (x % 2)
                    bayer.at<T>(y, x) = static_cast<T>(src.at<Vec3b>(y, x)[0]);
                else
                    bayer.at<T>(y, x) = static_cast<T>(src.at<Vec3b>(y, x)[2]);
            }
    }
}

TEST(Imgproc_ColorBayerVNG_Strict, regression)
{
    cvtest::TS* ts = cvtest::TS::ptr();
    const char pattern[][3] = { "bg", "gb", "rg", "gr" };
    const std::string image_name = "lena.png";
    const std::string parent_path = string(ts->get_data_path()) + "/cvtcolor_strict/";

    Mat src, dst, bayer, reference;
    std::string full_path = parent_path + image_name;
    src = imread(full_path, IMREAD_UNCHANGED);

    if ( src.empty() )
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_MISSING_TEST_DATA);
        ts->printf(cvtest::TS::SUMMARY, "No input image\n");
        ts->set_gtest_status();
        return;
    }

    for (int i = 0; i < 4; ++i)
    {
        calculateBayerPattern<uchar, CV_8U>(src, bayer, pattern[i]);
        CV_Assert(!bayer.empty() && bayer.type() == CV_8UC1);

        // calculating a dst image
        cvtColor(bayer, dst, CV_BayerBG2BGR_VNG + i);

        // reading a reference image
        full_path = parent_path + pattern[i] + image_name;
        reference = imread(full_path, IMREAD_UNCHANGED);
        if ( reference.empty() )
        {
            imwrite(full_path, dst);
            continue;
        }

        if (reference.depth() != dst.depth() || reference.channels() != dst.channels() ||
            reference.size() != dst.size())
        {
            std::cout << reference(Rect(0, 0, 5, 5)) << std::endl << std::endl << std::endl;
            ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
            ts->printf(cvtest::TS::SUMMARY, "\nReference channels: %d\n"
                "Actual channels: %d\n", reference.channels(), dst.channels());
            ts->printf(cvtest::TS::SUMMARY, "\nReference depth: %d\n"
                "Actual depth: %d\n", reference.depth(), dst.depth());
            ts->printf(cvtest::TS::SUMMARY, "\nReference rows: %d\n"
                "Actual rows: %d\n", reference.rows, dst.rows);
            ts->printf(cvtest::TS::SUMMARY, "\nReference cols: %d\n"
                "Actual cols: %d\n", reference.cols, dst.cols);
            ts->set_gtest_status();

            return;
        }

        Mat diff;
        absdiff(reference, dst, diff);

        int nonZero = countNonZero(diff.reshape(1) > 1);
        if (nonZero != 0)
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
            ts->printf(cvtest::TS::SUMMARY, "\nCount non zero in absdiff: %d\n", nonZero);
            ts->set_gtest_status();
            return;
        }
    }
}

static void getTestMatrix(Mat& src)
{
    Size ssize(1000, 1000);
    src.create(ssize, CV_32FC3);
    int szm = ssize.width - 1;
    float pi2 = 2 * 3.1415f;
    // Generate a pretty test image
    for (int i = 0; i < ssize.height; i++)
    {
        for (int j = 0; j < ssize.width; j++)
        {
            float b = (1 + cos((szm - i) * (szm - j) * pi2 / (10 * float(szm)))) / 2;
            float g = (1 + cos((szm - i) * j * pi2 / (10 * float(szm)))) / 2;
            float r = (1 + sin(i * j * pi2 / (10 * float(szm)))) / 2;

            // The following lines aren't necessary, but just to prove that
            // the BGR values all lie in [0,1]...
            if (b < 0) b = 0; else if (b > 1) b = 1;
            if (g < 0) g = 0; else if (g > 1) g = 1;
            if (r < 0) r = 0; else if (r > 1) r = 1;
            src.at<cv::Vec3f>(i, j) = cv::Vec3f(b, g, r);
        }
    }
}

static void validateResult(const Mat& reference, const Mat& actual, const Mat& src = Mat(), int mode = -1)
{
    cvtest::TS* ts = cvtest::TS::ptr();
    Size ssize = reference.size();

    int cn = reference.channels();
    ssize.width *= cn;
    bool next = true;

    for (int y = 0; y < ssize.height && next; ++y)
    {
        const float* rD = reference.ptr<float>(y);
        const float* D = actual.ptr<float>(y);
        for (int x = 0; x < ssize.width && next; ++x)
            if (fabs(rD[x] - D[x]) > 0.0001f)
            {
                next = false;
                ts->printf(cvtest::TS::SUMMARY, "Error in: (%d, %d)\n", x / cn,  y);
                ts->printf(cvtest::TS::SUMMARY, "Reference value: %f\n", rD[x]);
                ts->printf(cvtest::TS::SUMMARY, "Actual value: %f\n", D[x]);
                if (!src.empty())
                    ts->printf(cvtest::TS::SUMMARY, "Src value: %f\n", src.ptr<float>(y)[x]);
                ts->printf(cvtest::TS::SUMMARY, "Size: (%d, %d)\n", reference.rows, reference.cols);

                if (mode >= 0)
                {
                    cv::Mat lab;
                    cv::cvtColor(src, lab, mode);
                    std::cout << "lab: " << lab(cv::Rect(y, x / cn, 1, 1)) << std::endl;
                }
                std::cout << "src: " << src(cv::Rect(y, x / cn, 1, 1)) << std::endl;

                ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                ts->set_gtest_status();
            }
    }
}

TEST(Imgproc_ColorLab_Full, accuracy)
{
    Mat src;
    getTestMatrix(src);
    Size ssize = src.size();
    CV_Assert(ssize.width == ssize.height);

    RNG& rng = cvtest::TS::ptr()->get_rng();
    int blueInd = rng.uniform(0., 1.) > 0.5 ? 0 : 2;
    bool srgb = rng.uniform(0., 1.) > 0.5;

    // Convert test image to LAB
    cv::Mat lab;
    int forward_code = blueInd ? srgb ? CV_BGR2Lab : CV_LBGR2Lab : srgb ? CV_RGB2Lab : CV_LRGB2Lab;
    int inverse_code = blueInd ? srgb ? CV_Lab2BGR : CV_Lab2LBGR : srgb ? CV_Lab2RGB : CV_Lab2LRGB;
    cv::cvtColor(src, lab, forward_code);
    // Convert LAB image back to BGR(RGB)
    cv::Mat recons;
    cv::cvtColor(lab, recons, inverse_code);

    validateResult(src, recons, src, forward_code);
}

static void test_Bayer2RGB_EdgeAware_8u(const Mat& src, Mat& dst, int code)
{
    if (dst.empty())
        dst.create(src.size(), CV_MAKETYPE(src.depth(), 3));
    Size size = src.size();
    size.width -= 1;
    size.height -= 1;

    int dcn = dst.channels();
    CV_Assert(dcn == 3);

    int step = (int)src.step;
    const uchar* S = src.ptr<uchar>(1) + 1;
    uchar* D = dst.ptr<uchar>(1) + dcn;

    int start_with_green = code == CV_BayerGB2BGR_EA || code == CV_BayerGR2BGR_EA ? 1 : 0;
    int blue = code == CV_BayerGB2BGR_EA || code == CV_BayerBG2BGR_EA ? 1 : 0;

    for (int y = 1; y < size.height; ++y)
    {
        S = src.ptr<uchar>(y) + 1;
        D = dst.ptr<uchar>(y) + dcn;

        if (start_with_green)
        {
            for (int x = 1; x < size.width; x += 2, S += 2, D += 2*dcn)
            {
                // red
                D[0] = (S[-1] + S[1]) / 2;
                D[1] = S[0];
                D[2] = (S[-step] + S[step]) / 2;
                if (!blue)
                    std::swap(D[0], D[2]);
            }

            S = src.ptr<uchar>(y) + 2;
            D = dst.ptr<uchar>(y) + 2*dcn;

            for (int x = 2; x < size.width; x += 2, S += 2, D += 2*dcn)
            {
                // red
                D[0] = S[0];
                D[1] = (std::abs(S[-1] - S[1]) > std::abs(S[step] - S[-step]) ? (S[step] + S[-step] + 1) : (S[-1] + S[1] + 1)) / 2;
                D[2] = ((S[-step-1] + S[-step+1] + S[step-1] + S[step+1] + 2) / 4);
                if (!blue)
                    std::swap(D[0], D[2]);
            }
        }
        else
        {
            for (int x = 1; x < size.width; x += 2, S += 2, D += 2*dcn)
            {
                D[0] = S[0];
                D[1] = (std::abs(S[-1] - S[1]) > std::abs(S[step] - S[-step]) ? (S[step] + S[-step] + 1) : (S[-1] + S[1] + 1)) / 2;
                D[2] = ((S[-step-1] + S[-step+1] + S[step-1] + S[step+1] + 2) / 4);
                if (!blue)
                    std::swap(D[0], D[2]);
            }

            S = src.ptr<uchar>(y) + 2;
            D = dst.ptr<uchar>(y) + 2*dcn;

            for (int x = 2; x < size.width; x += 2, S += 2, D += 2*dcn)
            {
                D[0] = (S[-1] + S[1] + 1) / 2;
                D[1] = S[0];
                D[2] = (S[-step] + S[step] + 1) / 2;
                if (!blue)
                    std::swap(D[0], D[2]);
            }
        }

        D = dst.ptr<uchar>(y + 1) - dcn;
        for (int i = 0; i < dcn; ++i)
        {
            D[i] = D[-dcn + i];
            D[-static_cast<int>(dst.step)+dcn+i] = D[-static_cast<int>(dst.step)+(dcn<<1)+i];
        }

        start_with_green ^= 1;
        blue ^= 1;
    }

    ++size.width;
    uchar* firstRow = dst.ptr(), *lastRow = dst.ptr(size.height);
    size.width *= dcn;
    for (int x = 0; x < size.width; ++x)
    {
        firstRow[x] = firstRow[dst.step + x];
        lastRow[x] = lastRow[-static_cast<int>(dst.step)+x];
    }
}

template <typename T>
static void checkData(const Mat& actual, const Mat& reference, cvtest::TS* ts, const char* type,
    bool& next, const char* bayer_type)
{
    EXPECT_EQ(actual.size(), reference.size());
    EXPECT_EQ(actual.channels(), reference.channels());
    EXPECT_EQ(actual.depth(), reference.depth());

    Size size = reference.size();
    int dcn = reference.channels();
    size.width *= dcn;

    for (int y = 0; y < size.height && next; ++y)
    {
        const T* A = actual.ptr<T>(y);
        const T* R = reference.ptr<T>(y);

        for (int x = 0; x < size.width && next; ++x)
            if (std::abs(A[x] - R[x]) > 1)
            {
                #define SUM cvtest::TS::SUMMARY
                ts->printf(SUM, "\nReference value: %d\n", static_cast<int>(R[x]));
                ts->printf(SUM, "Actual value: %d\n", static_cast<int>(A[x]));
                ts->printf(SUM, "(y, x): (%d, %d)\n", y, x / reference.channels());
                ts->printf(SUM, "Channel pos: %d\n", x % reference.channels());
                ts->printf(SUM, "Pattern: %s\n", type);
                ts->printf(SUM, "Bayer image type: %s", bayer_type);
                #undef SUM

                Mat diff;
                absdiff(actual, reference, diff);
                EXPECT_EQ(countNonZero(diff.reshape(1) > 1), 0);

                ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                ts->set_gtest_status();

                next = false;
            }
    }
}

TEST(ImgProc_BayerEdgeAwareDemosaicing, accuracy)
{
    cvtest::TS* ts = cvtest::TS::ptr();
    const std::string image_name = "lena.png";
    const std::string parent_path = string(ts->get_data_path()) + "/cvtcolor_strict/";

    Mat src, bayer;
    std::string full_path = parent_path + image_name;
    src = imread(full_path, IMREAD_UNCHANGED);

    if (src.empty())
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_MISSING_TEST_DATA);
        ts->printf(cvtest::TS::SUMMARY, "No input image\n");
        ts->set_gtest_status();
        return;
    }

    /*
    COLOR_BayerBG2BGR_EA = 127,
    COLOR_BayerGB2BGR_EA = 128,
    COLOR_BayerRG2BGR_EA = 129,
    COLOR_BayerGR2BGR_EA = 130,
    */

    bool next = true;
    const char* types[] = { "bg", "gb", "rg", "gr" };
    for (int i = 0; i < 4 && next; ++i)
    {
        calculateBayerPattern<uchar, CV_8U>(src, bayer, types[i]);
        Mat reference;
        test_Bayer2RGB_EdgeAware_8u(bayer, reference, CV_BayerBG2BGR_EA + i);

        for (int t = 0; t <= 1; ++t)
        {
            if (t == 1)
                calculateBayerPattern<unsigned short int, CV_16U>(src, bayer, types[i]);

            CV_Assert(!bayer.empty() && (bayer.type() == CV_8UC1 || bayer.type() == CV_16UC1));

            Mat actual;
            cv::demosaicing(bayer, actual, CV_BayerBG2BGR_EA + i);

            if (t == 0)
                checkData<unsigned char>(actual, reference, ts, types[i], next, "CV_8U");
            else
            {
                Mat tmp;
                reference.convertTo(tmp, CV_16U);
                checkData<unsigned short int>(actual, tmp, ts, types[i], next, "CV_16U");
            }
        }
    }
}

TEST(ImgProc_Bayer2RGBA, accuracy)
{
    cvtest::TS* ts = cvtest::TS::ptr();
    Mat raw = imread(string(ts->get_data_path()) + "/cvtcolor/bayer_input.png", IMREAD_GRAYSCALE);
    Mat rgb, reference;

    CV_Assert(raw.channels() == 1);
    CV_Assert(raw.depth() == CV_8U);
    CV_Assert(!raw.empty());

    for (int code = CV_BayerBG2BGR; code <= CV_BayerGR2BGR; ++code)
    {
        cvtColor(raw, rgb, code);
        cvtColor(rgb, reference, CV_BGR2BGRA);

        Mat actual;
        cvtColor(raw, actual, code, 4);

        EXPECT_EQ(reference.size(), actual.size());
        EXPECT_EQ(reference.depth(), actual.depth());
        EXPECT_EQ(reference.channels(), actual.channels());

        Size ssize = raw.size();
        int cn = reference.channels();
        ssize.width *= cn;
        bool next = true;
        for (int y = 0; y < ssize.height && next; ++y)
        {
            const uchar* rD = reference.ptr<uchar>(y);
            const uchar* D = actual.ptr<uchar>(y);
            for (int x = 0; x < ssize.width && next; ++x)
                if (abs(rD[x] - D[x]) >= 1)
                {
                    next = false;
                    ts->printf(cvtest::TS::SUMMARY, "Error in: (%d, %d)\n", x / cn,  y);
                    ts->printf(cvtest::TS::SUMMARY, "Reference value: %d\n", rD[x]);
                    ts->printf(cvtest::TS::SUMMARY, "Actual value: %d\n", D[x]);
                    ts->printf(cvtest::TS::SUMMARY, "Src value: %d\n", raw.ptr<uchar>(y)[x]);
                    ts->printf(cvtest::TS::SUMMARY, "Size: (%d, %d)\n", reference.rows, reference.cols);

                    Mat diff;
                    absdiff(actual, reference, diff);
                    EXPECT_EQ(countNonZero(diff.reshape(1) > 1), 0);

                    ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                    ts->set_gtest_status();
                }
        }
    }
}

TEST(ImgProc_BGR2RGBA, regression_8696)
{
    Mat src(Size(80, 10), CV_8UC4);
    src.setTo(Scalar(255, 0, 200, 100));

    Mat dst;
    cvtColor(src, dst, COLOR_BGR2BGRA);

    EXPECT_DOUBLE_EQ(norm(dst - src, NORM_INF), 0.);
}

TEST(ImgProc_BGR2RGBA, 3ch24ch)
{
    Mat src(Size(80, 10), CV_8UC3);
    src.setTo(Scalar(200, 0, 200));

    Mat dst;
    cvtColor(src, dst, COLOR_BGR2BGRA);

    Mat expected(Size(80, 10), CV_8UC4);
    expected.setTo(Scalar(80, 0, 200, 255));

    EXPECT_DOUBLE_EQ(norm(expected - dst, NORM_INF), 0.);
}


//TODO: remove it later
/////////////
// computes cubic spline coefficients for a function: (xi=i, yi=f[i]), i=0..n
template<typename _Tp> static void splineBuild(const _Tp* f, int n, _Tp* tab)
{
    _Tp cn = 0;
    int i;
    tab[0] = tab[1] = (_Tp)0;

    for(i = 1; i < n-1; i++)
    {
        _Tp t = 3*(f[i+1] - 2*f[i] + f[i-1]);
        _Tp l = 1/(4 - tab[(i-1)*4]);
        tab[i*4] = l; tab[i*4+1] = (t - tab[(i-1)*4+1])*l;
    }

    for(i = n-1; i >= 0; i--)
    {
        _Tp c = tab[i*4+1] - tab[i*4]*cn;
        _Tp b = f[i+1] - f[i] - (cn + c*2)*(_Tp)0.3333333333333333;
        _Tp d = (cn - c)*(_Tp)0.3333333333333333;
        tab[i*4] = f[i]; tab[i*4+1] = b;
        tab[i*4+2] = c; tab[i*4+3] = d;
        cn = c;
    }
}

// interpolates value of a function at x, 0 <= x <= n using a cubic spline.
template<typename _Tp> static inline _Tp splineInterpolate(_Tp x, const _Tp* tab, int n)
{
    // don't touch this function without urgent need - some versions of gcc fail to inline it correctly
    int ix = std::min(std::max(int(x), 0), n-1);
    x -= ix;
    tab += ix*4;
    return ((tab[3]*x + tab[2])*x + tab[1])*x + tab[0];
}

#if CV_NEON
template<typename _Tp> static inline void splineInterpolate(float32x4_t& v_x, const _Tp* tab, int n)
{
    int32x4_t v_ix = vcvtq_s32_f32(vminq_f32(vmaxq_f32(v_x, vdupq_n_f32(0)), vdupq_n_f32(n - 1)));
    v_x = vsubq_f32(v_x, vcvtq_f32_s32(v_ix));
    v_ix = vshlq_n_s32(v_ix, 2);

    int CV_DECL_ALIGNED(16) ix[4];
    vst1q_s32(ix, v_ix);

    float32x4_t v_tab0 = vld1q_f32(tab + ix[0]);
    float32x4_t v_tab1 = vld1q_f32(tab + ix[1]);
    float32x4_t v_tab2 = vld1q_f32(tab + ix[2]);
    float32x4_t v_tab3 = vld1q_f32(tab + ix[3]);

    float32x4x2_t v01 = vtrnq_f32(v_tab0, v_tab1);
    float32x4x2_t v23 = vtrnq_f32(v_tab2, v_tab3);

    v_tab0 = vcombine_f32(vget_low_f32(v01.val[0]), vget_low_f32(v23.val[0]));
    v_tab1 = vcombine_f32(vget_low_f32(v01.val[1]), vget_low_f32(v23.val[1]));
    v_tab2 = vcombine_f32(vget_high_f32(v01.val[0]), vget_high_f32(v23.val[0]));
    v_tab3 = vcombine_f32(vget_high_f32(v01.val[1]), vget_high_f32(v23.val[1]));

    v_x = vmlaq_f32(v_tab0, vmlaq_f32(v_tab1, vmlaq_f32(v_tab2, v_tab3, v_x), v_x), v_x);
}
#elif CV_SSE2
template<typename _Tp> static inline void splineInterpolate(__m128& v_x, const _Tp* tab, int n)
{
    __m128i v_ix = _mm_cvttps_epi32(_mm_min_ps(_mm_max_ps(v_x, _mm_setzero_ps()), _mm_set1_ps(float(n - 1))));
    v_x = _mm_sub_ps(v_x, _mm_cvtepi32_ps(v_ix));
    v_ix = _mm_slli_epi32(v_ix, 2);

    int CV_DECL_ALIGNED(16) ix[4];
    _mm_store_si128((__m128i *)ix, v_ix);

    __m128 v_tab0 = _mm_loadu_ps(tab + ix[0]);
    __m128 v_tab1 = _mm_loadu_ps(tab + ix[1]);
    __m128 v_tab2 = _mm_loadu_ps(tab + ix[2]);
    __m128 v_tab3 = _mm_loadu_ps(tab + ix[3]);

    __m128 v_tmp0 = _mm_unpacklo_ps(v_tab0, v_tab1);
    __m128 v_tmp1 = _mm_unpacklo_ps(v_tab2, v_tab3);
    __m128 v_tmp2 = _mm_unpackhi_ps(v_tab0, v_tab1);
    __m128 v_tmp3 = _mm_unpackhi_ps(v_tab2, v_tab3);

    v_tab0 = _mm_shuffle_ps(v_tmp0, v_tmp1, 0x44);
    v_tab2 = _mm_shuffle_ps(v_tmp2, v_tmp3, 0x44);
    v_tab1 = _mm_shuffle_ps(v_tmp0, v_tmp1, 0xee);
    v_tab3 = _mm_shuffle_ps(v_tmp2, v_tmp3, 0xee);

    __m128 v_l = _mm_mul_ps(v_x, v_tab3);
    v_l = _mm_add_ps(v_l, v_tab2);
    v_l = _mm_mul_ps(v_l, v_x);
    v_l = _mm_add_ps(v_l, v_tab1);
    v_l = _mm_mul_ps(v_l, v_x);
    v_x = _mm_add_ps(v_l, v_tab0);
}
#endif

template<typename _Tp> struct ColorChannel
{
    typedef float worktype_f;
    static _Tp max() { return std::numeric_limits<_Tp>::max(); }
    static _Tp half() { return (_Tp)(max()/2 + 1); }
};

template<> struct ColorChannel<float>
{
    typedef float worktype_f;
    static float max() { return 1.f; }
    static float half() { return 0.5f; }
};

///////////

static const float sRGB2XYZ_D65[] =
{
    0.412453f, 0.357580f, 0.180423f,
    0.212671f, 0.715160f, 0.072169f,
    0.019334f, 0.119193f, 0.950227f
};

static const float XYZ2sRGB_D65[] =
{
    3.240479f, -1.53715f, -0.498535f,
    -0.969256f, 1.875991f, 0.041556f,
    0.055648f, -0.204043f, 1.057311f
};

enum
{
    yuv_shift = 14,
    xyz_shift = 12,
    R2Y = 4899, // == R2YF*16384
    G2Y = 9617, // == G2YF*16384
    B2Y = 1868, // == B2YF*16384
    BLOCK_SIZE = 256
};

#define  CV_DESCALE(x,n)     (((x) + (1 << ((n)-1))) >> (n))


///////////////////////////////////// RGB <-> L*a*b* /////////////////////////////////////

static const float D65[] = { 0.950456f, 1.f, 1.088754f };

enum { LAB_CBRT_TAB_SIZE = 1024, GAMMA_TAB_SIZE = 1024 };
static float LabCbrtTab[LAB_CBRT_TAB_SIZE*4];
static const float LabCbrtTabScale = LAB_CBRT_TAB_SIZE/1.5f;

static float sRGBGammaTab[GAMMA_TAB_SIZE*4], sRGBInvGammaTab[GAMMA_TAB_SIZE*4];
static const float GammaTabScale = (float)GAMMA_TAB_SIZE;

static ushort sRGBGammaTab_b[256], linearGammaTab_b[256];
enum { inv_gamma_shift = 12, INV_GAMMA_TAB_SIZE = (1 << inv_gamma_shift) };
static ushort sRGBInvGammaTab_b[INV_GAMMA_TAB_SIZE], linearInvGammaTab_b[INV_GAMMA_TAB_SIZE];
#undef lab_shift
#define lab_shift xyz_shift
#define gamma_shift 3
#define lab_shift2 (lab_shift + gamma_shift)
#define LAB_CBRT_TAB_SIZE_B (256*3/2*(1<<gamma_shift))
static ushort LabCbrtTab_b[LAB_CBRT_TAB_SIZE_B];

static bool enableTetraInterpolation = true;
//TODO: return it back
static bool enablePacked = false;
enum
{
    lab_lut_shift = 5,
    LAB_LUT_DIM = (1 << lab_lut_shift)+1,
    lab_base_shift = 14,
    LAB_BASE = (1 << lab_base_shift),
    trilinear_shift = 8 - lab_lut_shift + 1,
    TRILINEAR_BASE = (1 << trilinear_shift)
};
static int16_t Lab2RGBLUT_s16[LAB_LUT_DIM*LAB_LUT_DIM*LAB_LUT_DIM*3*8];
static int16_t RGB2LabLUT_s16[LAB_LUT_DIM*LAB_LUT_DIM*LAB_LUT_DIM*3*8];
static int16_t Lab2XYZLUT_s16[LAB_LUT_DIM*LAB_LUT_DIM*LAB_LUT_DIM*3*8];
static int16_t XYZ2LabLUT_s16[LAB_LUT_DIM*LAB_LUT_DIM*LAB_LUT_DIM*3*8];
static int16_t trilinearLUT[TRILINEAR_BASE*TRILINEAR_BASE*TRILINEAR_BASE*8];

//v*16384/255 ~~ v*16384/256 + v/4 if v in [0; 255]
static inline void div255(v_uint16x8& reg)
{
    reg = (reg << (lab_base_shift - 8)) + (reg >> 2);
}

template<int w, long long int d>
static inline long long int divConst(long long int v)
{
    int bits = 0; int dd = d;
    while(dd > 0)
    {
        dd = dd >> 1; bits++;
    }
    int b = bits - 1;
    long long int r = w + b;
    long long int pmod = (1l << r)%d;
    long long int f = (1l << r)/d;
    if(pmod)
    {
        if(pmod*2 > d)
        {
            v = (v * (f + 1) + (1l << (r - 1))) >> r;
        }
        else
        {
            v = ((v + 1) * f + (1l << (r - 1))) >> r;
        }
    }
    else
    {
        v = (v + (1l << (r - 1))) >> r;
    }
    return v;
}

#define clip(value) \
    value < 0.0f ? 0.0f : value > 1.0f ? 1.0f : value;

static inline float applyGamma(float x)
{
    return x <= 0.04045f ? x*(1.f/12.92f) : (float)std::pow((double)(x + 0.055)*(1./1.055), 2.4);
}

static inline float applyInvGamma(float x)
{
    return x <= 0.0031308 ? x*12.92f : (float)(1.055*std::pow((double)x, 1./2.4) - 0.055);
}

static inline void writeToLUT(int p, int q, int r, int16_t* LUT, uint16_t a, uint16_t b, uint16_t c)
{
    int idx = p*3 + q*LAB_LUT_DIM*3 + r*LAB_LUT_DIM*LAB_LUT_DIM*3;
    LUT[idx] = a; LUT[idx+1] = b; LUT[idx+2] = c;
}


static void initLabTabs()
{
    static bool initialized = false;
    if(!initialized)
    {
        float f[LAB_CBRT_TAB_SIZE+1], g[GAMMA_TAB_SIZE+1], ig[GAMMA_TAB_SIZE+1], scale = 1.f/LabCbrtTabScale;
        int i;
        for(i = 0; i <= LAB_CBRT_TAB_SIZE; i++)
        {
            float x = i*scale;
            f[i] = x < 0.008856f ? x*7.787f + 0.13793103448275862f : cvCbrt(x);
        }
        splineBuild(f, LAB_CBRT_TAB_SIZE, LabCbrtTab);

        scale = 1.f/GammaTabScale;
        for(i = 0; i <= GAMMA_TAB_SIZE; i++)
        {
            float x = i*scale;
            g[i] = applyGamma(x);
            ig[i] = applyInvGamma(x);
        }
        splineBuild(g, GAMMA_TAB_SIZE, sRGBGammaTab);
        splineBuild(ig, GAMMA_TAB_SIZE, sRGBInvGammaTab);

        for(i = 0; i < 256; i++)
        {
            float x = i*(1.f/255.f);
            sRGBGammaTab_b[i] = saturate_cast<ushort>(255.f*(1 << gamma_shift)*applyGamma(x));
            linearGammaTab_b[i] = (ushort)(i*(1 << gamma_shift));
        }
        float invScale = 1.f/INV_GAMMA_TAB_SIZE;
        for(i = 0; i < INV_GAMMA_TAB_SIZE; i++)
        {
            float x = i*invScale;
            sRGBInvGammaTab_b[i] = saturate_cast<ushort>(255.f*applyInvGamma(x));
            linearInvGammaTab_b[i] = (ushort)(255.f*x);
        }

        for(i = 0; i < LAB_CBRT_TAB_SIZE_B; i++)
        {
            float x = i*(1.f/(255.f*(1 << gamma_shift)));
            LabCbrtTab_b[i] = saturate_cast<ushort>((1 << lab_shift2)*(x < 0.008856f ? x*7.787f + 0.13793103448275862f : cvCbrt(x)));
        }

        if(enableTetraInterpolation)
        {
            static const float lThresh = 0.008856f * 903.3f;
            static const float fThresh = 7.787f * 0.008856f + 16.0f / 116.0f;
            static const float _a = 16.0f / 116.0f;

            const float* _whitept = D65;
            float coeffs[9];

            //Lab2RGB coeffs
            for(i = 0; i < 3; i++ )
            {
                coeffs[i+2*3] = XYZ2sRGB_D65[i]*_whitept[i];
                coeffs[i+1*3] = XYZ2sRGB_D65[i+3]*_whitept[i];
                coeffs[i+0*3] = XYZ2sRGB_D65[i+6]*_whitept[i];
            }
            float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
                  C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
                  C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];

            //RGB2Lab coeffs
            float scaleWhite[] = { 1.0f / _whitept[0], 1.0f, 1.0f / _whitept[2] };

            for(i = 0; i < 3; i++ )
            {
                int j = i * 3;
                coeffs[j + 2] = sRGB2XYZ_D65[j]     * scaleWhite[i];
                coeffs[j + 1] = sRGB2XYZ_D65[j + 1] * scaleWhite[i];
                coeffs[j + 0] = sRGB2XYZ_D65[j + 2] * scaleWhite[i];
            }

            float D0 = coeffs[0], D1 = coeffs[1], D2 = coeffs[2],
                  D3 = coeffs[3], D4 = coeffs[4], D5 = coeffs[5],
                  D6 = coeffs[6], D7 = coeffs[7], D8 = coeffs[8];

            AutoBuffer<int16_t> Lab2RGBprev(LAB_LUT_DIM*LAB_LUT_DIM*LAB_LUT_DIM*3);
            AutoBuffer<int16_t> RGB2Labprev(LAB_LUT_DIM*LAB_LUT_DIM*LAB_LUT_DIM*3);
            AutoBuffer<int16_t> Lab2XYZprev(LAB_LUT_DIM*LAB_LUT_DIM*LAB_LUT_DIM*3);
            AutoBuffer<int16_t> XYZ2Labprev(LAB_LUT_DIM*LAB_LUT_DIM*LAB_LUT_DIM*3);

            for(int p = 0; p < LAB_LUT_DIM; p++)
            {
                for(int q = 0; q < LAB_LUT_DIM; q++)
                {
                    for(int r = 0; r < LAB_LUT_DIM; r++)
                    {
                        //Lab 2 RGB LUTs building
                        float li = 100.0*p/(LAB_LUT_DIM-1);
                        float ai = 256.0*q/(LAB_LUT_DIM-1) - 128.0;
                        float bi = 256.0*r/(LAB_LUT_DIM-1) - 128.0;

                        float y, fy;
                        if (li <= lThresh)
                        {
                            y = li / 903.3f;
                            fy = 7.787f * y + 16.0f / 116.0f;
                        }
                        else
                        {
                            fy = (li + 16.0f) / 116.0f;
                            y = fy * fy * fy;
                        }

                        float fxz[] = { ai / 500.0f + fy, fy - bi / 200.0f };

                        for (int j = 0; j < 2; j++)
                            if (fxz[j] <= fThresh)
                                fxz[j] = (fxz[j] - 16.0f / 116.0f) / 7.787f;
                            else
                                fxz[j] = fxz[j] * fxz[j] * fxz[j];

                        float x = fxz[0], z = fxz[1];

                        //Lab(full range) => XYZ: x: [-0.0328753, 1.98139] y: [0, 1] z: [-0.0821883, 4.41094]
                        writeToLUT(p, q, r, Lab2XYZprev,
                                   cvRound(LAB_BASE*(x+0.04f)/2.03125f),
                                   cvRound(LAB_BASE*y),
                                   cvRound(LAB_BASE*(z+0.125f)/4.5f));

                        float ro = C0 * x + C1 * y + C2 * z;
                        float go = C3 * x + C4 * y + C5 * z;
                        float bo = C6 * x + C7 * y + C8 * z;
                        ro = clip(ro);
                        go = clip(go);
                        bo = clip(bo);

                        ro = applyInvGamma(ro);
                        go = applyInvGamma(go);
                        bo = applyInvGamma(bo);

                        writeToLUT(p, q, r, Lab2RGBprev,
                                   cvRound(LAB_BASE*ro),
                                   cvRound(LAB_BASE*go),
                                   cvRound(LAB_BASE*bo));

                        //RGB 2 Lab LUT building
                        float R = 1.0*p/(LAB_LUT_DIM-1);
                        float G = 1.0*q/(LAB_LUT_DIM-1);
                        float B = 1.0*r/(LAB_LUT_DIM-1);

                        R = applyGamma(R);
                        G = applyGamma(G);
                        B = applyGamma(B);

                        float X = R*D0 + G*D1 + B*D2;
                        float Y = R*D3 + G*D4 + B*D5;
                        float Z = R*D6 + G*D7 + B*D8;

                        float FX = X > 0.008856f ? std::pow(X, _1_3f) : (7.787f * X + _a);
                        float FY = Y > 0.008856f ? std::pow(Y, _1_3f) : (7.787f * Y + _a);
                        float FZ = Z > 0.008856f ? std::pow(Z, _1_3f) : (7.787f * Z + _a);

                        float L = Y > 0.008856f ? (116.f * FY - 16.f) : (903.3f * Y);
                        float a = 500.f * (FX - FY);
                        float b = 200.f * (FY - FZ);

                        writeToLUT(p, q, r, RGB2Labprev,
                                   cvRound(LAB_BASE*L/100.0f),
                                   cvRound(LAB_BASE*(a+128.0f)/256.0f),
                                   cvRound(LAB_BASE*(b+128.0f)/256.0f));

                        float xi = 1.0f*p/(LAB_LUT_DIM-1);
                        float yi = 1.0f*q/(LAB_LUT_DIM-1);
                        float zi = 1.0f*r/(LAB_LUT_DIM-1);

                        float iFX = xi > 0.008856f ? std::pow(xi, _1_3) : (7.787f * xi + _a);
                        float iFY = yi > 0.008856f ? std::pow(yi, _1_3) : (7.787f * yi + _a);
                        float iFZ = zi > 0.008856f ? std::pow(zi, _1_3) : (7.787f * zi + _a);

                        float iL = yi > 0.008856f ? (116.f * iFY - 16.f) : (903.3f * yi);
                        float ia = 500.f * (iFX - iFY);
                        float ib = 200.f * (iFY - iFZ);

                        writeToLUT(p, q, r, XYZ2Labprev,
                                   cvRound(LAB_BASE*iL/100.0f),
                                   cvRound(LAB_BASE*(ia+128.0f)/256.0f),
                                   cvRound(LAB_BASE*(ib+128.0f)/256.0f));
                    }
                }
            }
            for(int p = 0; p < LAB_LUT_DIM; p++)
            {
                for(int q = 0; q < LAB_LUT_DIM; q++)
                {
                    for(int r = 0; r < LAB_LUT_DIM; r++)
                    {
                        #define FILL(_p, _q, _r) \
                        do {\
                            int idxold = (p+(_p))*3 + (q+(_q))*LAB_LUT_DIM*3 + (r+(_r))*LAB_LUT_DIM*LAB_LUT_DIM*3;\
                            int idxnew = p*3*8 + q*LAB_LUT_DIM*3*8 + r*LAB_LUT_DIM*LAB_LUT_DIM*3*8+4*(_p)+2*(_q)+(_r);\
                            Lab2RGBLUT_s16[idxnew]    = Lab2RGBprev[idxold];\
                            Lab2RGBLUT_s16[idxnew+8]  = Lab2RGBprev[idxold+1];\
                            Lab2RGBLUT_s16[idxnew+16] = Lab2RGBprev[idxold+2];\
                            RGB2LabLUT_s16[idxnew]    = RGB2Labprev[idxold];\
                            RGB2LabLUT_s16[idxnew+8]  = RGB2Labprev[idxold+1];\
                            RGB2LabLUT_s16[idxnew+16] = RGB2Labprev[idxold+2];\
                            Lab2XYZLUT_s16[idxnew]    = Lab2XYZprev[idxold];\
                            Lab2XYZLUT_s16[idxnew+8]  = Lab2XYZprev[idxold+1];\
                            Lab2XYZLUT_s16[idxnew+16] = Lab2XYZprev[idxold+2];\
                            XYZ2LabLUT_s16[idxnew]    = XYZ2Labprev[idxold];\
                            XYZ2LabLUT_s16[idxnew+8]  = XYZ2Labprev[idxold+1];\
                            XYZ2LabLUT_s16[idxnew+16] = XYZ2Labprev[idxold+2];\
                        } while(0)

                        FILL(0, 0, 0);
                        FILL(0, 0, 1);
                        FILL(0, 1, 0);
                        FILL(0, 1, 1);
                        FILL(1, 0, 0);
                        FILL(1, 0, 1);
                        FILL(1, 1, 0);
                        FILL(1, 1, 1);

                        #undef FILL
                    }
                }
            }

            for(int p = 0; p < TRILINEAR_BASE; p++)
            {
                int16_t pp = TRILINEAR_BASE - p;
                for(int q = 0; q < TRILINEAR_BASE; q++)
                {
                    int16_t qq = TRILINEAR_BASE - q;
                    for(int r = 0; r < TRILINEAR_BASE; r++)
                    {
                        int16_t rr = TRILINEAR_BASE - r;
                        int16_t* w = &trilinearLUT[8*p + 8*TRILINEAR_BASE*q + 8*TRILINEAR_BASE*TRILINEAR_BASE*r];
                        w[0]  = pp * qq * rr; w[1]  = pp * qq * r ; w[2]  = pp * q  * rr; w[3]  = pp * q  * r ;
                        w[4]  = p  * qq * rr; w[5]  = p  * qq * r ; w[6]  = p  * q  * rr; w[7]  = p  * q  * r ;
                    }
                }
            }
        }

        initialized = true;
    }
}


// cx, cy, cz are in [0; LAB_BASE]
static inline void trilinearInterpolate(int cx, int cy, int cz, int16_t* LUT,
                                        int& a, int& b, int& c)
{
    //LUT idx of origin pt of cube
    int tx = cx >> (lab_base_shift - lab_lut_shift);
    int ty = cy >> (lab_base_shift - lab_lut_shift);
    int tz = cz >> (lab_base_shift - lab_lut_shift);

    int16_t* baseLUT = &LUT[3*8*tx + (3*8*LAB_LUT_DIM)*ty + (3*8*LAB_LUT_DIM*LAB_LUT_DIM)*tz];
    int aa[8], bb[8], cc[8];
    for(int i = 0; i < 8; i++)
    {
        aa[i] = baseLUT[i]; bb[i] = baseLUT[i+8]; cc[i] = baseLUT[i+16];
    }

    //x, y, z are [0; TRILINEAR_BASE)
    static const int bitMask = (1 << trilinear_shift) - 1;
    int x = (cx >> (lab_base_shift - 8 - 1)) & bitMask;
    int y = (cy >> (lab_base_shift - 8 - 1)) & bitMask;
    int z = (cz >> (lab_base_shift - 8 - 1)) & bitMask;

    int w[8];
    for(int i = 0; i < 8; i++)
    {
        w[i] = trilinearLUT[8*x + 8*TRILINEAR_BASE*y + 8*TRILINEAR_BASE*TRILINEAR_BASE*z + i];
    }

    a = aa[0]*w[0]+aa[1]*w[1]+aa[2]*w[2]+aa[3]*w[3]+aa[4]*w[4]+aa[5]*w[5]+aa[6]*w[6]+aa[7]*w[7];
    b = bb[0]*w[0]+bb[1]*w[1]+bb[2]*w[2]+bb[3]*w[3]+bb[4]*w[4]+bb[5]*w[5]+bb[6]*w[6]+bb[7]*w[7];
    c = cc[0]*w[0]+cc[1]*w[1]+cc[2]*w[2]+cc[3]*w[3]+cc[4]*w[4]+cc[5]*w[5]+cc[6]*w[6]+cc[7]*w[7];

    a = CV_DESCALE(a, trilinear_shift*3);
    b = CV_DESCALE(b, trilinear_shift*3);
    c = CV_DESCALE(c, trilinear_shift*3);
}


// 8 inValues are in [0; LAB_BASE]
static inline void trilinearPackedInterpolate(const v_uint16x8 inX, const v_uint16x8 inY, const v_uint16x8 inZ,
                                              const int16_t* LUT,
                                              v_uint16x8& outA, v_uint16x8& outB, v_uint16x8& outC)
{
    //LUT idx of origin pt of cube
    v_uint16x8 idxsX = inX >> (lab_base_shift - lab_lut_shift);
    v_uint16x8 idxsY = inY >> (lab_base_shift - lab_lut_shift);
    v_uint16x8 idxsZ = inZ >> (lab_base_shift - lab_lut_shift);

    //x, y, z are [0; TRILINEAR_BASE)
    const uint16_t bitMask = (1 << trilinear_shift) - 1;
    v_uint16x8 bitMaskReg = v_setall_u16(bitMask);
    v_uint16x8 fracX = (inX >> (lab_base_shift - 8 - 1)) & bitMaskReg;
    v_uint16x8 fracY = (inY >> (lab_base_shift - 8 - 1)) & bitMaskReg;
    v_uint16x8 fracZ = (inZ >> (lab_base_shift - 8 - 1)) & bitMaskReg;

    //load values to interpolate for pix0, pix1, .., pix7
    v_int16x8 a0, a1, a2, a3, a4, a5, a6, a7;
    v_int16x8 b0, b1, b2, b3, b4, b5, b6, b7;
    v_int16x8 c0, c1, c2, c3, c4, c5, c6, c7;

    v_uint32x4 addrDw0, addrDw1, addrDw10, addrDw11;
    v_mul_expand(v_setall_u16(3*8), idxsX, addrDw0, addrDw1);
    v_mul_expand(v_setall_u16(3*8*LAB_LUT_DIM), idxsY, addrDw10, addrDw11);
    addrDw0 += addrDw10; addrDw1 += addrDw11;
    v_mul_expand(v_setall_u16(3*8*LAB_LUT_DIM*LAB_LUT_DIM), idxsZ, addrDw10, addrDw11);
    addrDw0 += addrDw10; addrDw1 += addrDw11;

    uint32_t CV_DECL_ALIGNED(16) addrofs[8];
    v_store_aligned(addrofs, addrDw0);
    v_store_aligned(addrofs + 4, addrDw1);

    const int16_t* ptr;
#define LOAD_ABC(n) ptr = LUT + addrofs[n]; a##n = v_load(ptr); b##n = v_load(ptr + 8); c##n = v_load(ptr + 16)
    LOAD_ABC(0);
    LOAD_ABC(1);
    LOAD_ABC(2);
    LOAD_ABC(3);
    LOAD_ABC(4);
    LOAD_ABC(5);
    LOAD_ABC(6);
    LOAD_ABC(7);
#undef LOAD_ABC

    //interpolation weights for pix0, pix1, .., pix7
    v_int16x8 w0, w1, w2, w3, w4, w5, w6, w7;
    v_mul_expand(v_setall_u16(8), fracX, addrDw0, addrDw1);
    v_mul_expand(v_setall_u16(8*TRILINEAR_BASE), fracY, addrDw10, addrDw11);
    addrDw0 += addrDw10; addrDw1 += addrDw11;
    v_mul_expand(v_setall_u16(8*TRILINEAR_BASE*TRILINEAR_BASE), fracZ, addrDw10, addrDw11);
    addrDw0 += addrDw10; addrDw1 += addrDw11;

    v_store_aligned(addrofs, addrDw0);
    v_store_aligned(addrofs + 4, addrDw1);

#define LOAD_W(n) ptr = trilinearLUT + addrofs[n]; w##n = v_load(ptr)
    LOAD_W(0);
    LOAD_W(1);
    LOAD_W(2);
    LOAD_W(3);
    LOAD_W(4);
    LOAD_W(5);
    LOAD_W(6);
    LOAD_W(7);
#undef LOAD_W

    //outA = descale(v_reg<8>(sum(dot(ai, wi))))
    v_uint32x4 part0, part1;
#define DOT_SHIFT_PACK(l, ll) \
    part0 = v_uint32x4(v_reduce_sum(v_dotprod(l##0, w0)),\
                       v_reduce_sum(v_dotprod(l##1, w1)),\
                       v_reduce_sum(v_dotprod(l##2, w2)),\
                       v_reduce_sum(v_dotprod(l##3, w3)));\
    part1 = v_uint32x4(v_reduce_sum(v_dotprod(l##4, w4)),\
                       v_reduce_sum(v_dotprod(l##5, w5)),\
                       v_reduce_sum(v_dotprod(l##6, w6)),\
                       v_reduce_sum(v_dotprod(l##7, w7)));\
    part0 = (part0 + v_setall_u32(1 << (trilinear_shift*3-1))) >> (trilinear_shift*3);\
    part1 = (part1 + v_setall_u32(1 << (trilinear_shift*3-1))) >> (trilinear_shift*3);\
    (ll) = v_pack(part0, part1)

    DOT_SHIFT_PACK(a, outA);
    DOT_SHIFT_PACK(b, outB);
    DOT_SHIFT_PACK(c, outC);

#undef DOT_SHIFT_PACK
}


struct RGB2Lab_f
{
    typedef float channel_type;

    RGB2Lab_f(int _srccn, int _blueIdx, const float* _coeffs,
              const float* _whitept, bool _srgb)
    : srccn(_srccn), srgb(_srgb), blueIdx(_blueIdx)
    {
        volatile int _3 = 3;
        initLabTabs();

        useTetraInterpolation = (!_coeffs && !_whitept && srgb && enableTetraInterpolation);

        if (!_coeffs)
            _coeffs = sRGB2XYZ_D65;
        if (!_whitept)
            _whitept = D65;

        float scale[] = { 1.0f / _whitept[0], 1.0f, 1.0f / _whitept[2] };

        for( int i = 0; i < _3; i++ )
        {
            int j = i * 3;
            coeffs[j + (blueIdx ^ 2)] = _coeffs[j] * scale[i];
            coeffs[j + 1] = _coeffs[j + 1] * scale[i];
            coeffs[j + blueIdx] = _coeffs[j + 2] * scale[i];

            CV_Assert( coeffs[j] >= 0 && coeffs[j + 1] >= 0 && coeffs[j + 2] >= 0 &&
                       coeffs[j] + coeffs[j + 1] + coeffs[j + 2] < 1.5f*LabCbrtTabScale );
        }
    }

    void operator()(const float* src, float* dst, int n) const
    {
        int i, scn = srccn, bIdx = blueIdx;
        float gscale = GammaTabScale;
        const float* gammaTab = srgb ? sRGBGammaTab : 0;
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        n *= 3;

        i = 0;
        if(useTetraInterpolation)
        {
            //TODO: add packed version

            for(; i < n; i += 3, src += scn)
            {
                float R = clip(src[bIdx]);
                float G = clip(src[1]);
                float B = clip(src[bIdx^2]);

                //don't use XYZ table in RGB2Lab conversion
                int iR = R*LAB_BASE, iG = G*LAB_BASE, iB = B*LAB_BASE, iL, ia, ib;
                trilinearInterpolate(iR, iG, iB, RGB2LabLUT_s16, iL, ia, ib);
                float L = iL*1.0f/LAB_BASE, a = ia*1.0f/LAB_BASE, b = ib*1.0f/LAB_BASE;

                dst[i] = L*100.0f;
                dst[i + 1] = a*256.0f - 128.0f;
                dst[i + 2] = b*256.0f - 128.0f;
            }
        }

        static const float _a = 16.0f / 116.0f;
        for (; i < n; i += 3, src += scn )
        {
            float R = clip(src[0]);
            float G = clip(src[1]);
            float B = clip(src[2]);

            if (gammaTab)
            {
                R = splineInterpolate(R * gscale, gammaTab, GAMMA_TAB_SIZE);
                G = splineInterpolate(G * gscale, gammaTab, GAMMA_TAB_SIZE);
                B = splineInterpolate(B * gscale, gammaTab, GAMMA_TAB_SIZE);
            }
            float X = R*C0 + G*C1 + B*C2;
            float Y = R*C3 + G*C4 + B*C5;
            float Z = R*C6 + G*C7 + B*C8;

            float FX = X > 0.008856f ? std::pow(X, _1_3f) : (7.787f * X + _a);
            float FY = Y > 0.008856f ? std::pow(Y, _1_3f) : (7.787f * Y + _a);
            float FZ = Z > 0.008856f ? std::pow(Z, _1_3f) : (7.787f * Z + _a);

            float L = Y > 0.008856f ? (116.f * FY - 16.f) : (903.3f * Y);
            float a = 500.f * (FX - FY);
            float b = 200.f * (FY - FZ);

            dst[i] = L;
            dst[i + 1] = a;
            dst[i + 2] = b;
        }
    }

    int srccn;
    float coeffs[9];
    bool srgb;
    bool useTetraInterpolation;
    int blueIdx;
};


struct Lab2RGB_f
{
    typedef float channel_type;

    Lab2RGB_f( int _dstcn, int _blueIdx, const float* _coeffs,
              const float* _whitept, bool _srgb )
    : dstcn(_dstcn), srgb(_srgb), blueIdx(_blueIdx)
    {
        initLabTabs();

        useTetraInterpolation = (!_coeffs && !_whitept && srgb && enableTetraInterpolation);

        if(!_coeffs)
            _coeffs = XYZ2sRGB_D65;
        if(!_whitept)
            _whitept = D65;

        for( int i = 0; i < 3; i++ )
        {
            coeffs[i+(blueIdx^2)*3] = _coeffs[i]*_whitept[i];
            coeffs[i+3] = _coeffs[i+3]*_whitept[i];
            coeffs[i+blueIdx*3] = _coeffs[i+6]*_whitept[i];
        }

        lThresh = 0.008856f * 903.3f;
        fThresh = 7.787f * 0.008856f + 16.0f / 116.0f;
        #if CV_SSE2
        haveSIMD = checkHardwareSupport(CV_CPU_SSE2);
        #endif
    }

    #if CV_SSE2
    void process(__m128& v_li0, __m128& v_li1, __m128& v_ai0,
                 __m128& v_ai1, __m128& v_bi0, __m128& v_bi1) const
    {
        __m128 v_y00 = _mm_mul_ps(v_li0, _mm_set1_ps(1.0f/903.3f));
        __m128 v_y01 = _mm_mul_ps(v_li1, _mm_set1_ps(1.0f/903.3f));
        __m128 v_fy00 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(7.787f), v_y00), _mm_set1_ps(16.0f/116.0f));
        __m128 v_fy01 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(7.787f), v_y01), _mm_set1_ps(16.0f/116.0f));

        __m128 v_fy10 = _mm_mul_ps(_mm_add_ps(v_li0, _mm_set1_ps(16.0f)), _mm_set1_ps(1.0f/116.0f));
        __m128 v_fy11 = _mm_mul_ps(_mm_add_ps(v_li1, _mm_set1_ps(16.0f)), _mm_set1_ps(1.0f/116.0f));
        __m128 v_y10 = _mm_mul_ps(_mm_mul_ps(v_fy10, v_fy10), v_fy10);
        __m128 v_y11 = _mm_mul_ps(_mm_mul_ps(v_fy11, v_fy11), v_fy11);

        __m128 v_cmpli0 = _mm_cmple_ps(v_li0, _mm_set1_ps(lThresh));
        __m128 v_cmpli1 = _mm_cmple_ps(v_li1, _mm_set1_ps(lThresh));
        v_y00 = _mm_and_ps(v_cmpli0, v_y00);
        v_y01 = _mm_and_ps(v_cmpli1, v_y01);
        v_fy00 = _mm_and_ps(v_cmpli0, v_fy00);
        v_fy01 = _mm_and_ps(v_cmpli1, v_fy01);
        v_y10 = _mm_andnot_ps(v_cmpli0, v_y10);
        v_y11 = _mm_andnot_ps(v_cmpli1, v_y11);
        v_fy10 = _mm_andnot_ps(v_cmpli0, v_fy10);
        v_fy11 = _mm_andnot_ps(v_cmpli1, v_fy11);
        __m128 v_y0 = _mm_or_ps(v_y00, v_y10);
        __m128 v_y1 = _mm_or_ps(v_y01, v_y11);
        __m128 v_fy0 = _mm_or_ps(v_fy00, v_fy10);
        __m128 v_fy1 = _mm_or_ps(v_fy01, v_fy11);

        __m128 v_fxz00 = _mm_add_ps(v_fy0, _mm_mul_ps(v_ai0, _mm_set1_ps(0.002f)));
        __m128 v_fxz01 = _mm_add_ps(v_fy1, _mm_mul_ps(v_ai1, _mm_set1_ps(0.002f)));
        __m128 v_fxz10 = _mm_sub_ps(v_fy0, _mm_mul_ps(v_bi0, _mm_set1_ps(0.005f)));
        __m128 v_fxz11 = _mm_sub_ps(v_fy1, _mm_mul_ps(v_bi1, _mm_set1_ps(0.005f)));

        __m128 v_fxz000 = _mm_mul_ps(_mm_sub_ps(v_fxz00, _mm_set1_ps(16.0f/116.0f)), _mm_set1_ps(1.0f/7.787f));
        __m128 v_fxz001 = _mm_mul_ps(_mm_sub_ps(v_fxz01, _mm_set1_ps(16.0f/116.0f)), _mm_set1_ps(1.0f/7.787f));
        __m128 v_fxz010 = _mm_mul_ps(_mm_sub_ps(v_fxz10, _mm_set1_ps(16.0f/116.0f)), _mm_set1_ps(1.0f/7.787f));
        __m128 v_fxz011 = _mm_mul_ps(_mm_sub_ps(v_fxz11, _mm_set1_ps(16.0f/116.0f)), _mm_set1_ps(1.0f/7.787f));

        __m128 v_fxz100 = _mm_mul_ps(_mm_mul_ps(v_fxz00, v_fxz00), v_fxz00);
        __m128 v_fxz101 = _mm_mul_ps(_mm_mul_ps(v_fxz01, v_fxz01), v_fxz01);
        __m128 v_fxz110 = _mm_mul_ps(_mm_mul_ps(v_fxz10, v_fxz10), v_fxz10);
        __m128 v_fxz111 = _mm_mul_ps(_mm_mul_ps(v_fxz11, v_fxz11), v_fxz11);

        __m128 v_cmpfxz00 = _mm_cmple_ps(v_fxz00, _mm_set1_ps(fThresh));
        __m128 v_cmpfxz01 = _mm_cmple_ps(v_fxz01, _mm_set1_ps(fThresh));
        __m128 v_cmpfxz10 = _mm_cmple_ps(v_fxz10, _mm_set1_ps(fThresh));
        __m128 v_cmpfxz11 = _mm_cmple_ps(v_fxz11, _mm_set1_ps(fThresh));
        v_fxz000 = _mm_and_ps(v_cmpfxz00, v_fxz000);
        v_fxz001 = _mm_and_ps(v_cmpfxz01, v_fxz001);
        v_fxz010 = _mm_and_ps(v_cmpfxz10, v_fxz010);
        v_fxz011 = _mm_and_ps(v_cmpfxz11, v_fxz011);
        v_fxz100 = _mm_andnot_ps(v_cmpfxz00, v_fxz100);
        v_fxz101 = _mm_andnot_ps(v_cmpfxz01, v_fxz101);
        v_fxz110 = _mm_andnot_ps(v_cmpfxz10, v_fxz110);
        v_fxz111 = _mm_andnot_ps(v_cmpfxz11, v_fxz111);
        __m128 v_x0 = _mm_or_ps(v_fxz000, v_fxz100);
        __m128 v_x1 = _mm_or_ps(v_fxz001, v_fxz101);
        __m128 v_z0 = _mm_or_ps(v_fxz010, v_fxz110);
        __m128 v_z1 = _mm_or_ps(v_fxz011, v_fxz111);

        __m128 v_ro0 = _mm_mul_ps(_mm_set1_ps(coeffs[0]), v_x0);
        __m128 v_ro1 = _mm_mul_ps(_mm_set1_ps(coeffs[0]), v_x1);
        __m128 v_go0 = _mm_mul_ps(_mm_set1_ps(coeffs[3]), v_x0);
        __m128 v_go1 = _mm_mul_ps(_mm_set1_ps(coeffs[3]), v_x1);
        __m128 v_bo0 = _mm_mul_ps(_mm_set1_ps(coeffs[6]), v_x0);
        __m128 v_bo1 = _mm_mul_ps(_mm_set1_ps(coeffs[6]), v_x1);
        v_ro0 = _mm_add_ps(v_ro0, _mm_mul_ps(_mm_set1_ps(coeffs[1]), v_y0));
        v_ro1 = _mm_add_ps(v_ro1, _mm_mul_ps(_mm_set1_ps(coeffs[1]), v_y1));
        v_go0 = _mm_add_ps(v_go0, _mm_mul_ps(_mm_set1_ps(coeffs[4]), v_y0));
        v_go1 = _mm_add_ps(v_go1, _mm_mul_ps(_mm_set1_ps(coeffs[4]), v_y1));
        v_bo0 = _mm_add_ps(v_bo0, _mm_mul_ps(_mm_set1_ps(coeffs[7]), v_y0));
        v_bo1 = _mm_add_ps(v_bo1, _mm_mul_ps(_mm_set1_ps(coeffs[7]), v_y1));
        v_ro0 = _mm_add_ps(v_ro0, _mm_mul_ps(_mm_set1_ps(coeffs[2]), v_z0));
        v_ro1 = _mm_add_ps(v_ro1, _mm_mul_ps(_mm_set1_ps(coeffs[2]), v_z1));
        v_go0 = _mm_add_ps(v_go0, _mm_mul_ps(_mm_set1_ps(coeffs[5]), v_z0));
        v_go1 = _mm_add_ps(v_go1, _mm_mul_ps(_mm_set1_ps(coeffs[5]), v_z1));
        v_bo0 = _mm_add_ps(v_bo0, _mm_mul_ps(_mm_set1_ps(coeffs[8]), v_z0));
        v_bo1 = _mm_add_ps(v_bo1, _mm_mul_ps(_mm_set1_ps(coeffs[8]), v_z1));

        v_li0 = _mm_min_ps(_mm_max_ps(v_ro0, _mm_setzero_ps()), _mm_set1_ps(1.0f));
        v_li1 = _mm_min_ps(_mm_max_ps(v_ro1, _mm_setzero_ps()), _mm_set1_ps(1.0f));
        v_ai0 = _mm_min_ps(_mm_max_ps(v_go0, _mm_setzero_ps()), _mm_set1_ps(1.0f));
        v_ai1 = _mm_min_ps(_mm_max_ps(v_go1, _mm_setzero_ps()), _mm_set1_ps(1.0f));
        v_bi0 = _mm_min_ps(_mm_max_ps(v_bo0, _mm_setzero_ps()), _mm_set1_ps(1.0f));
        v_bi1 = _mm_min_ps(_mm_max_ps(v_bo1, _mm_setzero_ps()), _mm_set1_ps(1.0f));
    }
    #endif

    void operator()(const float* src, float* dst, int n) const
    {
        int i = 0, dcn = dstcn, bIdx = blueIdx;
        const float* gammaTab = srgb ? sRGBInvGammaTab : 0;
        float gscale = GammaTabScale;
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
        C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
        C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        float alpha = ColorChannel<float>::max();
        n *= 3;

        if(useTetraInterpolation)
        {
            //TODO: insert packed version

            for (; i < n; i += 3, dst += dcn)
            {
                float li = src[i];
                float ai = src[i + 1];
                float bi = src[i + 2];

                //use XYZ table in Lab2RGB conversion
                int il = li/100.0f*LAB_BASE, ia = (ai+128.0f)/256.0f*LAB_BASE, ib = (bi+128.0f)/256.0f*LAB_BASE;
                int iX, iY, iZ;
                trilinearInterpolate(il, ia, ib, Lab2XYZLUT_s16, iX, iY, iZ);

                float ro, go, bo;
                //Lab(full range) => XYZ: x: [-0.0328753, 1.98139] y: [0, 1] z: [-0.0821883, 4.41094]
                float x = (iX*1.0f/LAB_BASE)*2.5f-0.125f, y = (iY*1.0f/LAB_BASE), z = (iZ*1.0f/LAB_BASE)*4.5f-0.125f;
                ro = C0 * x + C1 * y + C2 * z;
                go = C3 * x + C4 * y + C5 * z;
                bo = C6 * x + C7 * y + C8 * z;
                ro = clip(ro);
                go = clip(go);
                bo = clip(bo);

                if (gammaTab)
                {
                    ro = splineInterpolate(ro * gscale, gammaTab, GAMMA_TAB_SIZE);
                    go = splineInterpolate(go * gscale, gammaTab, GAMMA_TAB_SIZE);
                    bo = splineInterpolate(bo * gscale, gammaTab, GAMMA_TAB_SIZE);
                }

                dst[bIdx] = ro, dst[1] = go, dst[bIdx^2] = bo;
                if( dcn == 4 )
                    dst[3] = alpha;
            }
        }

        #if CV_SSE2
        if (haveSIMD)
        {
            for (; i <= n - 24; i += 24, dst += dcn * 8)
            {
                __m128 v_li0 = _mm_loadu_ps(src + i +  0);
                __m128 v_li1 = _mm_loadu_ps(src + i +  4);
                __m128 v_ai0 = _mm_loadu_ps(src + i +  8);
                __m128 v_ai1 = _mm_loadu_ps(src + i + 12);
                __m128 v_bi0 = _mm_loadu_ps(src + i + 16);
                __m128 v_bi1 = _mm_loadu_ps(src + i + 20);

                _mm_deinterleave_ps(v_li0, v_li1, v_ai0, v_ai1, v_bi0, v_bi1);

                process(v_li0, v_li1, v_ai0, v_ai1, v_bi0, v_bi1);

                if (gammaTab)
                {
                    __m128 v_gscale = _mm_set1_ps(gscale);
                    v_li0 = _mm_mul_ps(v_li0, v_gscale);
                    v_li1 = _mm_mul_ps(v_li1, v_gscale);
                    v_ai0 = _mm_mul_ps(v_ai0, v_gscale);
                    v_ai1 = _mm_mul_ps(v_ai1, v_gscale);
                    v_bi0 = _mm_mul_ps(v_bi0, v_gscale);
                    v_bi1 = _mm_mul_ps(v_bi1, v_gscale);

                    splineInterpolate(v_li0, gammaTab, GAMMA_TAB_SIZE);
                    splineInterpolate(v_li1, gammaTab, GAMMA_TAB_SIZE);
                    splineInterpolate(v_ai0, gammaTab, GAMMA_TAB_SIZE);
                    splineInterpolate(v_ai1, gammaTab, GAMMA_TAB_SIZE);
                    splineInterpolate(v_bi0, gammaTab, GAMMA_TAB_SIZE);
                    splineInterpolate(v_bi1, gammaTab, GAMMA_TAB_SIZE);
                }

                if( dcn == 4 )
                {
                    __m128 v_a0 = _mm_set1_ps(alpha);
                    __m128 v_a1 = _mm_set1_ps(alpha);
                    _mm_interleave_ps(v_li0, v_li1, v_ai0, v_ai1, v_bi0, v_bi1, v_a0, v_a1);

                    _mm_storeu_ps(dst +  0, v_li0);
                    _mm_storeu_ps(dst +  4, v_li1);
                    _mm_storeu_ps(dst +  8, v_ai0);
                    _mm_storeu_ps(dst + 12, v_ai1);
                    _mm_storeu_ps(dst + 16, v_bi0);
                    _mm_storeu_ps(dst + 20, v_bi1);
                    _mm_storeu_ps(dst + 24, v_a0);
                    _mm_storeu_ps(dst + 28, v_a1);
                }
                else
                {
                    _mm_interleave_ps(v_li0, v_li1, v_ai0, v_ai1, v_bi0, v_bi1);

                    _mm_storeu_ps(dst +  0, v_li0);
                    _mm_storeu_ps(dst +  4, v_li1);
                    _mm_storeu_ps(dst +  8, v_ai0);
                    _mm_storeu_ps(dst + 12, v_ai1);
                    _mm_storeu_ps(dst + 16, v_bi0);
                    _mm_storeu_ps(dst + 20, v_bi1);
                }
            }
        }
        #endif
        for (; i < n; i += 3, dst += dcn)
        {
            float li = src[i];
            float ai = src[i + 1];
            float bi = src[i + 2];

            float y, fy;
            if (li <= lThresh)
            {
                y = li / 903.3f;
                fy = 7.787f * y + 16.0f / 116.0f;
            }
            else
            {
                fy = (li + 16.0f) / 116.0f;
                y = fy * fy * fy;
            }

            float fxz[] = { ai / 500.0f + fy, fy - bi / 200.0f };

            for (int j = 0; j < 2; j++)
                if (fxz[j] <= fThresh)
                    fxz[j] = (fxz[j] - 16.0f / 116.0f) / 7.787f;
                else
                    fxz[j] = fxz[j] * fxz[j] * fxz[j];

            float x = fxz[0], z = fxz[1];
            float ro = C0 * x + C1 * y + C2 * z;
            float go = C3 * x + C4 * y + C5 * z;
            float bo = C6 * x + C7 * y + C8 * z;
            ro = clip(ro);
            go = clip(go);
            bo = clip(bo);

            //TODO: return it back
            if (gammaTab)
            {
                ro = splineInterpolate(ro * gscale, gammaTab, GAMMA_TAB_SIZE);
                go = splineInterpolate(go * gscale, gammaTab, GAMMA_TAB_SIZE);
                bo = splineInterpolate(bo * gscale, gammaTab, GAMMA_TAB_SIZE);
            }

            dst[0] = ro, dst[1] = go, dst[2] = bo;
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }

    int dstcn;
    float coeffs[9];
    bool srgb;
    float lThresh;
    float fThresh;
    #if CV_SSE2
    bool haveSIMD;
    #endif
    int blueIdx;
    bool useTetraInterpolation;
};


struct RGB2Lab_b
{
    typedef uchar channel_type;

    RGB2Lab_b(int _srccn, int _blueIdx, const float* _coeffs,
              const float* _whitept, bool _srgb)
    : srccn(_srccn), srgb(_srgb), blueIdx(_blueIdx)
    {
        static volatile int _3 = 3;
        initLabTabs();

        useTetraInterpolation = (!_coeffs && !_whitept && srgb && enableTetraInterpolation);

        if (!_coeffs)
            _coeffs = sRGB2XYZ_D65;
        if (!_whitept)
            _whitept = D65;

        float scale[] =
        {
            (1 << lab_shift)/_whitept[0],
            (float)(1 << lab_shift),
            (1 << lab_shift)/_whitept[2]
        };

        for( int i = 0; i < _3; i++ )
        {
            coeffs[i*3+(blueIdx^2)] = cvRound(_coeffs[i*3]*scale[i]);
            coeffs[i*3+1] = cvRound(_coeffs[i*3+1]*scale[i]);
            coeffs[i*3+blueIdx] = cvRound(_coeffs[i*3+2]*scale[i]);

            CV_Assert( coeffs[i] >= 0 && coeffs[i*3+1] >= 0 && coeffs[i*3+2] >= 0 &&
                      coeffs[i*3] + coeffs[i*3+1] + coeffs[i*3+2] < 2*(1 << lab_shift) );
        }
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        const int Lscale = (116*255+50)/100;
        const int Lshift = -((16*255*(1 << lab_shift2) + 50)/100);
        const ushort* tab = srgb ? sRGBGammaTab_b : linearGammaTab_b;
        int bIdx = blueIdx;
        int i, scn = srccn;
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
            C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
            C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        n *= 3;

        i = 0;
        if(useTetraInterpolation)
        {
            for(; enablePacked && (i <= n-3*8*2); i += 3*8*2, src += scn*8*2)
            {
                v_uint16x8 rvec0, rvec1, gvec0, gvec1, bvec0, bvec1, dummy;
                v_uint8x16 u8r, u8g, u8b, u8a;
                if(scn == 3)
                {
                    v_load_deinterleave(src, u8r, u8g, u8b);
                }
                else if(scn == 4)
                {
                    v_load_deinterleave(src, u8r, u8g, u8b, u8a);
                }
                v_expand(u8r, rvec0, rvec1);
                v_expand(u8g, gvec0, gvec1);
                v_expand(u8b, bvec0, bvec1);

                if(bIdx > 0)
                {
                    dummy = rvec0; rvec0 = bvec0; bvec0 = dummy;
                    dummy = rvec1; rvec1 = bvec1; bvec1 = dummy;
                }

                // (r, g, b) *= (LAB_BASE/255);
                div255(rvec0); div255(rvec1);
                div255(gvec0); div255(gvec1);
                div255(bvec0); div255(bvec1);

                //don't use XYZ table for RGB2Lab
                v_uint16x8 l_vec0, l_vec1, a_vec0, a_vec1, b_vec0, b_vec1;
                trilinearPackedInterpolate(rvec0, gvec0, bvec0, RGB2LabLUT_s16, l_vec0, a_vec0, b_vec0);
                trilinearPackedInterpolate(rvec1, gvec1, bvec1, RGB2LabLUT_s16, l_vec1, a_vec1, b_vec1);

                // l = l*255/LAB_BASE
                v_uint16x8 scaleReg = v_setall_u16(255); v_uint32x4 dw0, dw1;
                v_mul_expand(l_vec0, scaleReg, dw0, dw1);
                dw0 = dw0 >> lab_base_shift; dw1 = dw1 >> lab_base_shift;
                l_vec0 = v_pack(dw0, dw1);
                v_mul_expand(l_vec1, scaleReg, dw0, dw1);
                dw0 = dw0 >> lab_base_shift; dw1 = dw1 >> lab_base_shift;
                l_vec1 = v_pack(dw0, dw1);

                // (a, b) /= (LAB_BASE/256);
                a_vec0 = a_vec0 >> (lab_base_shift - 8); a_vec1 = a_vec1 >> (lab_base_shift - 8);
                b_vec0 = b_vec0 >> (lab_base_shift - 8); b_vec1 = b_vec1 >> (lab_base_shift - 8);

                v_uint8x16 u8_l = v_pack(l_vec0, l_vec1);
                v_uint8x16 u8_a = v_pack(a_vec0, a_vec1);
                v_uint8x16 u8_b = v_pack(b_vec0, b_vec1);

                v_store_interleave(dst + i, u8_l, u8_a, u8_b);
            }

            for(; i < n; i += 3, src += scn)
            {
                int R = src[bIdx], G = src[1], B = src[bIdx^2];

                //don't use XYZ table for RGB2Lab
                R = R*LAB_BASE/255, G = G*LAB_BASE/255, B = B*LAB_BASE/255;

                int L, a, b;
                trilinearInterpolate(R, G, B, RGB2LabLUT_s16, L, a, b);

                dst[i] = saturate_cast<uchar>(L*255/LAB_BASE);
                dst[i+1] = saturate_cast<uchar>(a/(LAB_BASE/256));
                dst[i+2] = saturate_cast<uchar>(b/(LAB_BASE/256));
            }

        }

        for(; i < n; i += 3, src += scn )
        {
            int R = tab[src[0]], G = tab[src[1]], B = tab[src[2]];
            int fX = LabCbrtTab_b[CV_DESCALE(R*C0 + G*C1 + B*C2, lab_shift)];
            int fY = LabCbrtTab_b[CV_DESCALE(R*C3 + G*C4 + B*C5, lab_shift)];
            int fZ = LabCbrtTab_b[CV_DESCALE(R*C6 + G*C7 + B*C8, lab_shift)];

            int L = CV_DESCALE( Lscale*fY + Lshift, lab_shift2 );
            int a = CV_DESCALE( 500*(fX - fY) + 128*(1 << lab_shift2), lab_shift2 );
            int b = CV_DESCALE( 200*(fY - fZ) + 128*(1 << lab_shift2), lab_shift2 );

            dst[i] = saturate_cast<uchar>(L);
            dst[i+1] = saturate_cast<uchar>(a);
            dst[i+2] = saturate_cast<uchar>(b);
        }
    }

    int srccn;
    int coeffs[9];
    bool srgb;
    int blueIdx;
    bool useTetraInterpolation;
};


//TODO: rewrite it to work w/o scaling values
struct Lab2RGB_b
{
    typedef uchar channel_type;

    Lab2RGB_b( int _dstcn, int _blueIdx, const float* _coeffs,
               const float* _whitept, bool _srgb )
    : dstcn(_dstcn), cvt(3, _blueIdx, _coeffs, _whitept, _srgb ), blueIdx(_blueIdx), srgb(_srgb)
    {
        useTetraInterpolation = (!_coeffs && !_whitept && _srgb && enableTetraInterpolation);

        if(!_coeffs)
            _coeffs = XYZ2sRGB_D65;
        if(!_whitept)
            _whitept = D65;

        for(int i = 0; i < 3; i++)
        {
            coeffs[i+(blueIdx^2*3)] = cvRound((1 << lab_shift)*_coeffs[i]*_whitept[i]);
            coeffs[i+3] = cvRound((1 << lab_shift)*_coeffs[i+3]*_whitept[i]);
            coeffs[i+blueIdx*3] = cvRound((1 << lab_shift)*_coeffs[i+6]*_whitept[i]);
        }

        #if CV_NEON
        v_scale_inv = vdupq_n_f32(100.f/255.f);
        v_scale = vdupq_n_f32(255.f);
        v_alpha = vdup_n_u8(ColorChannel<uchar>::max());
        v_128 = vdupq_n_f32(128.0f);
        #elif CV_SSE2
        v_scale = _mm_set1_ps(255.f);
        v_alpha = _mm_set1_ps(ColorChannel<uchar>::max());
        v_zero = _mm_setzero_si128();
        haveSIMD = checkHardwareSupport(CV_CPU_SSE2);
        #endif
    }

    #if CV_SSE2
    // 16s x 8
    void process(__m128i v_r, __m128i v_g, __m128i v_b,
                 const __m128& v_coeffs_, const __m128& v_res_,
                 float * buf) const
    {
        __m128 v_r0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_r, v_zero));
        __m128 v_g0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_g, v_zero));
        __m128 v_b0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_b, v_zero));

        __m128 v_r1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_r, v_zero));
        __m128 v_g1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_g, v_zero));
        __m128 v_b1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_b, v_zero));

        __m128 v_coeffs = v_coeffs_;
        __m128 v_res = v_res_;

        v_r0 = _mm_sub_ps(_mm_mul_ps(v_r0, v_coeffs), v_res);
        v_g1 = _mm_sub_ps(_mm_mul_ps(v_g1, v_coeffs), v_res);

        v_coeffs = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v_coeffs), 0x49));
        v_res = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v_res), 0x49));

        v_r1 = _mm_sub_ps(_mm_mul_ps(v_r1, v_coeffs), v_res);
        v_b0 = _mm_sub_ps(_mm_mul_ps(v_b0, v_coeffs), v_res);

        v_coeffs = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v_coeffs), 0x49));
        v_res = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v_res), 0x49));

        v_g0 = _mm_sub_ps(_mm_mul_ps(v_g0, v_coeffs), v_res);
        v_b1 = _mm_sub_ps(_mm_mul_ps(v_b1, v_coeffs), v_res);

        _mm_store_ps(buf, v_r0);
        _mm_store_ps(buf + 4, v_r1);
        _mm_store_ps(buf + 8, v_g0);
        _mm_store_ps(buf + 12, v_g1);
        _mm_store_ps(buf + 16, v_b0);
        _mm_store_ps(buf + 20, v_b1);
    }
    #endif

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i, j, dcn = dstcn;
        int bIdx = blueIdx;
        const ushort* tab = srgb ? sRGBInvGammaTab_b : linearInvGammaTab_b;
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
        C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
        C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        uchar alpha = ColorChannel<uchar>::max();
        float CV_DECL_ALIGNED(16) buf[3*BLOCK_SIZE];
        #if CV_SSE2
        __m128 v_coeffs = _mm_set_ps(100.f/255.f, 1.f, 1.f, 100.f/255.f);
        __m128 v_res = _mm_set_ps(0.f, 128.f, 128.f, 0.f);
        #endif

        i = 0;
        if(useTetraInterpolation)
        {
            for(; enablePacked && (i <= n*3-3*8*2); i += 3*8*2, dst += dcn*8*2)
            {
                v_uint8x16 u8l, u8a, u8b;
                v_uint16x8 lvec0, lvec1, avec0, avec1, bvec0, bvec1;
                v_load_deinterleave(src + i, u8l, u8a, u8b);
                v_expand(u8l, lvec0, lvec1);
                v_expand(u8a, avec0, avec1);
                v_expand(u8b, bvec0, bvec1);

                //l = l*LAB_BASE/255
                div255(lvec0); div255(lvec1);
                //(a, b) * = LAB_BASE/256
                avec0 = avec0 << (lab_base_shift - 8); avec1 = avec1 << (lab_base_shift - 8);
                bvec0 = bvec0 << (lab_base_shift - 8); bvec1 = bvec1 << (lab_base_shift - 8);

                //use XYZ table when doing Lab2RGB conversion
                v_uint16x8 x_vec0, x_vec1, y_vec0, y_vec1, z_vec0, z_vec1;
                trilinearPackedInterpolate(lvec0, avec0, bvec0, Lab2XYZLUT_s16, x_vec0, y_vec0, z_vec0);
                trilinearPackedInterpolate(lvec1, avec1, bvec1, Lab2XYZLUT_s16, x_vec1, y_vec1, z_vec1);

                v_int16x8 x_vec0s(x_vec0.val), x_vec1s(x_vec1.val);
                v_int16x8 y_vec0s(y_vec0.val), y_vec1s(y_vec1.val);
                v_int16x8 z_vec0s(z_vec0.val), z_vec1s(z_vec1.val);
                v_int32x4 xdw_00, xdw_01, xdw_10, xdw_11;
                v_int32x4 ydw_00, ydw_01, ydw_10, ydw_11;
                v_int32x4 zdw_00, zdw_01, zdw_10, zdw_11;
                v_expand(x_vec0s, xdw_00, xdw_01); v_expand(x_vec1s, xdw_10, xdw_11);
                v_expand(y_vec0s, ydw_00, ydw_01); v_expand(y_vec1s, ydw_10, ydw_11);
                v_expand(z_vec0s, zdw_00, zdw_01); v_expand(z_vec1s, zdw_10, zdw_11);

                //Lab(full range) => XYZ: x: [-0.0328753, 1.98139] y: [0, 1] z: [-0.0821883, 4.41094]
                //int x = (ro*20-LAB_BASE)/8, y = go, z = (bo*36-LAB_BASE)/8;
                v_int32x4 baseReg = v_setall_s32(LAB_BASE), mulReg = v_setall_s32(20);
                xdw_00 = (xdw_00*mulReg - baseReg) >> 3; xdw_01 = (xdw_01*mulReg - baseReg) >> 3;
                xdw_10 = (xdw_10*mulReg - baseReg) >> 3; xdw_11 = (xdw_11*mulReg - baseReg) >> 3;
                mulReg = v_setall_s32(36);
                zdw_00 = (zdw_00*mulReg - baseReg) >> 3; zdw_01 = (zdw_01*mulReg - baseReg) >> 3;
                zdw_10 = (zdw_10*mulReg - baseReg) >> 3; zdw_11 = (zdw_11*mulReg - baseReg) >> 3;

                const int shift = lab_shift+(lab_base_shift-inv_gamma_shift);
                /*
                    ro = CV_DESCALE(C0 * x + C1 * y + C2 * z, shift);
                    go = CV_DESCALE(C3 * x + C4 * y + C5 * z, shift);
                    bo = CV_DESCALE(C6 * x + C7 * y + C8 * z, shift);
                */
                v_int32x4 rdw_00, rdw_01, rdw_10, rdw_11;
                v_int32x4 gdw_00, gdw_01, gdw_10, gdw_11;
                v_int32x4 bdw_00, bdw_01, bdw_10, bdw_11;
                v_int32x4 descaleReg = v_setall_s32(1 << ((shift)-1));
                v_int32x4 c0reg = v_setall_s32(C0), c1reg = v_setall_s32(C1), c2reg = v_setall_s32(C2),
                          c3reg = v_setall_s32(C3), c4reg = v_setall_s32(C4), c5reg = v_setall_s32(C5),
                          c6reg = v_setall_s32(C6), c7reg = v_setall_s32(C7), c8reg = v_setall_s32(C8);

                #define MUL_XYZ(l, xn, yn, zn) \
                    l##dw_00 = (xdw_00*c##xn##reg + ydw_00*c##yn##reg + zdw_00*c##zn##reg + descaleReg) >> shift;\
                    l##dw_01 = (xdw_01*c##xn##reg + ydw_01*c##yn##reg + zdw_01*c##zn##reg + descaleReg) >> shift;\
                    l##dw_10 = (xdw_10*c##xn##reg + ydw_10*c##yn##reg + zdw_10*c##zn##reg + descaleReg) >> shift;\
                    l##dw_11 = (xdw_11*c##xn##reg + ydw_11*c##yn##reg + zdw_11*c##zn##reg + descaleReg) >> shift

                MUL_XYZ(r, 0, 1, 2);
                MUL_XYZ(g, 3, 4, 5);
                MUL_XYZ(b, 6, 7, 8);
                #undef MUL_XYZ

                v_int16x8 r_vec0s, r_vec1s, g_vec0s, g_vec1s, b_vec0s, b_vec1s;

                r_vec0s = v_pack(rdw_00, rdw_01); r_vec1s = v_pack(rdw_10, rdw_11);
                g_vec0s = v_pack(gdw_00, gdw_01); g_vec1s = v_pack(gdw_10, gdw_11);
                b_vec0s = v_pack(bdw_00, bdw_01); b_vec1s = v_pack(bdw_10, bdw_11);

                //limit indices in table
                v_int16x8 tabsz = v_setall_s16((int)INV_GAMMA_TAB_SIZE-1);
                #define CLAMP(r) (r) = v_max(v_setzero_s16(), v_min((r), tabsz))
                CLAMP(r_vec0s); CLAMP(r_vec1s);
                CLAMP(g_vec0s); CLAMP(g_vec1s);
                CLAMP(b_vec0s); CLAMP(b_vec1s);
                #undef CLAMP

                v_uint16x8 r_vec0(r_vec0s.val), r_vec1(r_vec1s.val);
                v_uint16x8 g_vec0(g_vec0s.val), g_vec1(g_vec1s.val);
                v_uint16x8 b_vec0(b_vec0s.val), b_vec1(b_vec1s.val);

                //ro = tab[ro]; go = tab[go]; bo = tab[bo];
                uint16_t CV_DECL_ALIGNED(16) shifts[8];
                #define GAMMA_TAB_SUBST(reg) \
                v_store_aligned(shifts, (reg));\
                (reg) = v_uint16x8(tab[shifts[0]],\
                                   tab[shifts[1]],\
                                   tab[shifts[2]],\
                                   tab[shifts[3]],\
                                   tab[shifts[4]],\
                                   tab[shifts[5]],\
                                   tab[shifts[6]],\
                                   tab[shifts[7]])

                GAMMA_TAB_SUBST(r_vec0); GAMMA_TAB_SUBST(r_vec1);
                GAMMA_TAB_SUBST(g_vec0); GAMMA_TAB_SUBST(g_vec1);
                GAMMA_TAB_SUBST(b_vec0); GAMMA_TAB_SUBST(b_vec1);

                #undef GAMMA_TAB_SUBST

                v_uint8x16 u8_b = v_pack(b_vec0, b_vec1);
                v_uint8x16 u8_g = v_pack(g_vec0, g_vec1);
                v_uint8x16 u8_r = v_pack(r_vec0, r_vec1);

                v_uint8x16 dummy;
                if(bIdx == 0)
                {
                    dummy = u8_r; u8_r = u8_b; u8_b = dummy;
                }

                if(dcn == 4)
                {
                    v_uint8x16 u8_alpha = v_setall_u8(alpha);
                    v_store_interleave(dst, u8_b, u8_g, u8_r, u8_alpha);
                }
                else
                {
                    v_store_interleave(dst, u8_b, u8_g, u8_r);
                }
            }

            for (; i < n*3; i += 3, dst += dcn)
            {
                int ro, go, bo, x, y, z;
                static const int BASE = (1 << 14);
                static const int lThresh = 0.008856f * 903.3f * (long long int)BASE/100;
                static const int fThresh = (7.787f * 0.008856f + 16.0f / 116.0f)*BASE;
                int L = src[i + 0]*BASE/255;
                int a = (src[i + 1] - 128)*BASE/256;
                int b = (src[i + 2] - 128)*BASE/256;

                int ify;
                const int base16_116 = BASE*16/116 + 1;
                //const bool natDiv = true;
                int x1, y1, z1, x2, y2, z2;
                //if(natDiv)
                {
                    if(L <= lThresh)
                    {
                        //yy = li / 903.3f;
                        //y = L*100/903.3f;
                        y = divConst<14, 9033>((long long int)L*1000);

                        //fy = 7.787f * yy + 16.0f / 116.0f;
                        ify = base16_116 + y*8 - y*213/1000;
                    }
                    else
                    {
                        //fy = (li + 16.0f) / 116.0f;
                        ify = L*100/116 + base16_116;

                        //yy = fy * fy * fy;
                        y = ify*ify/BASE*ify/BASE;
                    }

                    //float fxz[] = { ai / 500.0f + fy, fy - bi / 200.0f };
                    int adiv, bdiv;
                    //adiv = a*256/500, bdiv = b*256/200;
                    const int bits = 24;
                    adiv = divConst<bits, 500>((long long int)a*256);
                    bdiv = divConst<bits, 200>((long long int)b*256);

                    int ifxz[] = {ify + adiv, ify - bdiv};
                    for(int k = 0; k < 2; k++)
                    {
                        int& v = ifxz[k];
                        if(v <= fThresh)
                        {
                            //fxz[k] = (fxz[k] - 16.0f / 116.0f) / 7.787f;
                            v = v*1000/7787 - BASE*16/116*1000/7787;
                        }
                        else
                        {
                            //fxz[k] = fxz[k] * fxz[k] * fxz[k];
                            v = v*v/BASE*v/BASE;
                        }
                    }
                    x1 = ifxz[0]/(BASE/LAB_BASE); y1 = y/(BASE/LAB_BASE); z1 = ifxz[1]/(BASE/LAB_BASE);
                }
                //else
                {
                    if(L <= lThresh)
                    {
                        //yy = li / 903.3f;
                        //y = L*100/903.3f;

                        y = divConst<14, 9033>((long long int)L*1000);

                        //fy = 7.787f * yy + 16.0f / 116.0f;
                        ify = base16_116 + y*8 - divConst<14, 1000>((long long int)y*213);
                    }
                    else
                    {
                        //fy = (li + 16.0f) / 116.0f;
                        ify = divConst<20, 116>((long long int)L*100) + base16_116;

                        //yy = fy * fy * fy;
                        y = ify*ify/BASE*ify/BASE;
                    }

                    //float fxz[] = { ai / 500.0f + fy, fy - bi / 200.0f };
                    int adiv, bdiv;
                    const int bits = 24;
                    adiv = divConst<bits, 500>((long long int)a*256);
                    bdiv = divConst<bits, 200>((long long int)b*256);

                    int ifxz[] = {ify + adiv, ify - bdiv};
                    for(int k = 0; k < 2; k++)
                    {
                        int& v = ifxz[k];
                        if(v <= fThresh)
                        {
                            //fxz[k] = (fxz[k] - 16.0f / 116.0f) / 7.787f;
                            v = divConst<14, 7787>((long long int)v*1000) - BASE*16/116*1000/7787;
                        }
                        else
                        {
                            //fxz[k] = fxz[k] * fxz[k] * fxz[k];
                            v = v*v/BASE*v/BASE;
                        }
                    }
                    x2 = ifxz[0]/(BASE/LAB_BASE); y2 = y/(BASE/LAB_BASE); z2 = ifxz[1]/(BASE/LAB_BASE);
                }
                x = x1, y = y1, z = z1;

                const int shift = lab_shift+(lab_base_shift-inv_gamma_shift);
                ro = CV_DESCALE(C0 * x + C1 * y + C2 * z, shift);
                go = CV_DESCALE(C3 * x + C4 * y + C5 * z, shift);
                bo = CV_DESCALE(C6 * x + C7 * y + C8 * z, shift);

                ro = max(0, min((int)INV_GAMMA_TAB_SIZE-1, ro));
                go = max(0, min((int)INV_GAMMA_TAB_SIZE-1, go));
                bo = max(0, min((int)INV_GAMMA_TAB_SIZE-1, bo));

                ro = tab[ro];
                go = tab[go];
                bo = tab[bo];

                dst[bIdx^2] = saturate_cast<uchar>(bo);
                dst[1] = saturate_cast<uchar>(go);
                dst[bIdx] = saturate_cast<uchar>(ro);
                if( dcn == 4 )
                    dst[3] = alpha;
            }
        }

        for(; i < n; i += BLOCK_SIZE, src += BLOCK_SIZE*3 )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);
            j = 0;

            #if CV_NEON
            for ( ; j <= (dn - 8) * 3; j += 24)
            {
                uint8x8x3_t v_src = vld3_u8(src + j);
                uint16x8_t v_t0 = vmovl_u8(v_src.val[0]),
                           v_t1 = vmovl_u8(v_src.val[1]),
                           v_t2 = vmovl_u8(v_src.val[2]);

                float32x4x3_t v_dst;
                v_dst.val[0] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t0))), v_scale_inv);
                v_dst.val[1] = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t1))), v_128);
                v_dst.val[2] = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t2))), v_128);
                vst3q_f32(buf + j, v_dst);

                v_dst.val[0] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t0))), v_scale_inv);
                v_dst.val[1] = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t1))), v_128);
                v_dst.val[2] = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t2))), v_128);
                vst3q_f32(buf + j + 12, v_dst);
            }
            #elif CV_SSE2
            if (haveSIMD)
            {
                for ( ; j <= (dn - 8) * 3; j += 24)
                {
                    __m128i v_src0 = _mm_loadu_si128((__m128i const *)(src + j));
                    __m128i v_src1 = _mm_loadl_epi64((__m128i const *)(src + j + 16));

                    process(_mm_unpacklo_epi8(v_src0, v_zero),
                            _mm_unpackhi_epi8(v_src0, v_zero),
                            _mm_unpacklo_epi8(v_src1, v_zero),
                            v_coeffs, v_res,
                            buf + j);
                }
            }
            #endif

            for( ; j < dn*3; j += 3 )
            {
                buf[j] = src[j]*(100.f/255.f);
                buf[j+1] = (float)(src[j+1] - 128);
                buf[j+2] = (float)(src[j+2] - 128);
            }
            cvt(buf, buf, dn);
            j = 0;

            #if CV_NEON
            for ( ; j <= (dn - 8) * 3; j += 24, dst += dcn * 8)
            {
                float32x4x3_t v_src0 = vld3q_f32(buf + j), v_src1 = vld3q_f32(buf + j + 12);
                uint8x8_t v_dst0 = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src0.val[0], v_scale))),
                                                           vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src1.val[0], v_scale)))));
                uint8x8_t v_dst1 = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src0.val[1], v_scale))),
                                                           vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src1.val[1], v_scale)))));
                uint8x8_t v_dst2 = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src0.val[2], v_scale))),
                                                           vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src1.val[2], v_scale)))));

                if (dcn == 4)
                {
                    uint8x8x4_t v_dst;
                    v_dst.val[0] = v_dst0;
                    v_dst.val[1] = v_dst1;
                    v_dst.val[2] = v_dst2;
                    v_dst.val[3] = v_alpha;
                    vst4_u8(dst, v_dst);
                }
                else
                {
                    uint8x8x3_t v_dst;
                    v_dst.val[0] = v_dst0;
                    v_dst.val[1] = v_dst1;
                    v_dst.val[2] = v_dst2;
                    vst3_u8(dst, v_dst);
                }
            }
            #elif CV_SSE2
            if (dcn == 3 && haveSIMD)
            {
                for ( ; j <= (dn * 3 - 16); j += 16, dst += 16)
                {
                    __m128 v_src0 = _mm_mul_ps(_mm_load_ps(buf + j), v_scale);
                    __m128 v_src1 = _mm_mul_ps(_mm_load_ps(buf + j + 4), v_scale);
                    __m128 v_src2 = _mm_mul_ps(_mm_load_ps(buf + j + 8), v_scale);
                    __m128 v_src3 = _mm_mul_ps(_mm_load_ps(buf + j + 12), v_scale);

                    __m128i v_dst0 = _mm_packs_epi32(_mm_cvtps_epi32(v_src0),
                                                     _mm_cvtps_epi32(v_src1));
                    __m128i v_dst1 = _mm_packs_epi32(_mm_cvtps_epi32(v_src2),
                                                     _mm_cvtps_epi32(v_src3));

                    _mm_storeu_si128((__m128i *)dst, _mm_packus_epi16(v_dst0, v_dst1));
                }

                int jr = j % 3;
                if (jr)
                    dst -= jr, j -= jr;
            }
            else if (dcn == 4 && haveSIMD)
            {
                for ( ; j <= (dn * 3 - 12); j += 12, dst += 16)
                {
                    __m128 v_buf0 = _mm_mul_ps(_mm_load_ps(buf + j), v_scale);
                    __m128 v_buf1 = _mm_mul_ps(_mm_load_ps(buf + j + 4), v_scale);
                    __m128 v_buf2 = _mm_mul_ps(_mm_load_ps(buf + j + 8), v_scale);

                    __m128 v_ba0 = _mm_unpackhi_ps(v_buf0, v_alpha);
                    __m128 v_ba1 = _mm_unpacklo_ps(v_buf2, v_alpha);

                    __m128i v_src0 = _mm_cvtps_epi32(_mm_shuffle_ps(v_buf0, v_ba0, 0x44));
                    __m128i v_src1 = _mm_shuffle_epi32(_mm_cvtps_epi32(_mm_shuffle_ps(v_ba0, v_buf1, 0x4e)), 0x78);
                    __m128i v_src2 = _mm_cvtps_epi32(_mm_shuffle_ps(v_buf1, v_ba1, 0x4e));
                    __m128i v_src3 = _mm_shuffle_epi32(_mm_cvtps_epi32(_mm_shuffle_ps(v_ba1, v_buf2, 0xee)), 0x78);

                    __m128i v_dst0 = _mm_packs_epi32(v_src0, v_src1);
                    __m128i v_dst1 = _mm_packs_epi32(v_src2, v_src3);

                    _mm_storeu_si128((__m128i *)dst, _mm_packus_epi16(v_dst0, v_dst1));
                }

                int jr = j % 3;
                if (jr)
                    dst -= jr, j -= jr;
            }
            #endif

            for( ; j < dn*3; j += 3, dst += dcn )
            {
                dst[0] = saturate_cast<uchar>(buf[j]*255.f);
                dst[1] = saturate_cast<uchar>(buf[j+1]*255.f);
                dst[2] = saturate_cast<uchar>(buf[j+2]*255.f);
                if( dcn == 4 )
                    dst[3] = alpha;
            }
        }
    }

    int dstcn;
    Lab2RGB_f cvt;

    #if CV_NEON
    float32x4_t v_scale, v_scale_inv, v_128;
    uint8x8_t v_alpha;
    #elif CV_SSE2
    __m128 v_scale;
    __m128 v_alpha;
    __m128i v_zero;
    bool haveSIMD;
    #endif
    bool useTetraInterpolation;
    int blueIdx;
    float coeffs[9];
    bool srgb;
};

#undef clip
#undef DIV255

/////////////

TEST(ImgProc_Color, LabCheckWorking)
{
    cv::setUseOptimized(false);

    //settings
    #define INT_DATA 1
    #define TO_BGR 1
    const bool randomFill = true;

    enableTetraInterpolation = true;
    Lab2RGB_f interToBgr(3, 0, 0, 0, true);
    RGB2Lab_f interToLab(3, 0, 0, 0, true);
    Lab2RGB_b interToBgr_b(3, 0, 0, 0, true);
    RGB2Lab_b interToLab_b(3, 0, 0, 0, true);
    enableTetraInterpolation = false;
    Lab2RGB_f goldToBgr(3, 0, 0, 0, true);
    RGB2Lab_f goldToLab(3, 0, 0, 0, true);
    Lab2RGB_b goldToBgr_b(3, 0, 0, 0, true);
    RGB2Lab_b goldToLab_b(3, 0, 0, 0, true);

    char bgrChannels[3] = {'b', 'g', 'r'};
    char labChannels[3] = {'l', 'a', 'b'};
    char* channel = TO_BGR ? bgrChannels : labChannels;

    int nPerfIters = 100;

    string dir = "/home/savuor/logs/ocv/lab_precision/" + string(TO_BGR ? "lab2bgr/" : "rgb2lab/");

    const size_t pSize = 256+1;
    Mat  mGold(pSize, pSize, CV_32FC3);
    Mat   mSrc(pSize, pSize, CV_32FC3);
    Mat mInter(pSize, pSize, CV_32FC3);
    Mat   mBackGold(pSize, pSize, CV_32FC3);
    Mat  mBackInter(pSize, pSize, CV_32FC3);

    if(INT_DATA)
    {
        mGold  = Mat(pSize, pSize, CV_8UC3);
        mSrc   = Mat(pSize, pSize, CV_8UC3);
        mInter = Mat(pSize, pSize, CV_8UC3);
        mBackGold  = Mat(pSize, pSize, CV_8UC3);
        mBackInter = Mat(pSize, pSize, CV_8UC3);
    }

    Scalar vmean, vdev;
    std::vector<Mat> chDiff, chInter;
    double vmin[3], vmax[3]; Point minPt[3], maxPt[3];
    double maxMaxError[4] = {-100, -100, -100, -100};
    double times[4] = {0, 0, 0, 0};
    int count = 0;

    int blue = 0, l = 0;
#if TO_BGR
    for(; l < 100+1; l++)
#else
    for(; blue < 256+1; blue++)
#endif
    {
        for(size_t p = 0; p < pSize; p++)
        {
            float* pRow   = mSrc.ptr<float>(p);
            uchar* pRow_b = mSrc.ptr<uchar>(p);
            for(size_t q = 0; q < pSize; q++)
            {
                if(INT_DATA)
                {
                    if(TO_BGR)
                    {
                        //Lab
                        pRow_b[3*q + 0] = l*255/100;
                        pRow_b[3*q + 1] = q;
                        pRow_b[3*q + 2] = p;
                    }
                    else
                    {
                        //BGR
                        pRow_b[3*q + 0] = blue;
                        pRow_b[3*q + 1] = q;
                        pRow_b[3*q + 2] = p;
                    }
                }
                else
                {
                    if(TO_BGR)
                    {
                        //Lab
                        pRow[3*q + 0] = 1.0f*l;
                        pRow[3*q + 1] = 256.0f*q/(pSize-1)-128.0f;
                        pRow[3*q + 2] = 256.0f*p/(pSize-1)-128.0f;
                    }
                    else
                    {
                        //BGR
                        pRow[3*q + 0] = 1.0f*blue/(pSize-1);
                        pRow[3*q + 1] = 1.0f*q/(pSize-1);
                        pRow[3*q + 2] = 1.0f*p/(pSize-1);
                    }
                }

            }
        }

        for(size_t p = 0; p < pSize; p++)
        {
            float* pSrc   =   mSrc.ptr<float>(p);
            float* pGold  =  mGold.ptr<float>(p);
            float* pInter = mInter.ptr<float>(p);
            float* pBackGold  = mBackGold.ptr<float>(p);
            float* pBackInter = mBackInter.ptr<float>(p);

            uchar* pSrc_b   =   mSrc.ptr<uchar>(p);
            uchar* pGold_b  =  mGold.ptr<uchar>(p);
            uchar* pInter_b = mInter.ptr<uchar>(p);
            uchar* pBackGold_b  = mBackGold.ptr<uchar>(p);
            uchar* pBackInter_b = mBackInter.ptr<uchar>(p);
            if(INT_DATA)
            {
                if(TO_BGR)
                {
                    interToBgr_b(pSrc_b, pInter_b, pSize);
                    goldToBgr_b(pSrc_b, pGold_b, pSize);

                    interToLab_b(pInter_b, pBackInter_b, pSize);
                    goldToLab_b(pGold_b, pBackGold_b, pSize);
                }
                else
                {
                    interToLab_b(pSrc_b, pInter_b, pSize);
                    goldToLab_b(pSrc_b, pGold_b, pSize);

                    interToBgr_b(pInter_b, pBackInter_b, pSize);
                    goldToBgr_b(pGold_b, pBackGold_b, pSize);
                }
            }
            else
            {
                if(TO_BGR)
                {
                    interToBgr(pSrc, pInter, pSize);
                    goldToBgr(pSrc, pGold, pSize);

                    interToLab(pInter, pBackInter, pSize);
                    goldToLab(pGold, pBackGold, pSize);
                }
                else
                {
                    interToLab(pSrc, pInter, pSize);
                    goldToLab(pSrc, pGold, pSize);

                    interToBgr(pInter, pBackInter, pSize);
                    goldToBgr(pGold, pBackGold, pSize);
                }
            }
        }

        std::cout << (TO_BGR ? l : blue) << ":" << endl;

        Mat diff = abs(mGold-mInter);
        meanStdDev(diff, vmean, vdev);
        std::cout << "absdiff: mean " << vmean << " stddev " << vdev << std::endl;
        split(diff, chDiff);
        split(mInter, chInter);
        for(int c = 0; c < 3; c++)
        {
            minMaxLoc(chDiff[c], &vmin[c], &vmax[c],
                      &minPt[c], &maxPt[c]);
            std::cout << " ch "  << channel[c];
            std::cout << " max " << vmax[c] << " at " << maxPt[c];
            maxMaxError[0] = max(maxMaxError[0], vmax[c]);
        }
        std::cout << std::endl;

        Mat backGoldDiff = abs(mBackGold - mSrc);
        meanStdDev(backGoldDiff, vmean, vdev);
        std::cout << "backGoldDiff: mean " << vmean << " stddev " << vdev << std::endl;
        split(backGoldDiff, chDiff);
        for(int c = 0; c < 3; c++)
        {
            minMaxLoc(chDiff[c], &vmin[c], &vmax[c],
                      &minPt[c], &maxPt[c]);
            std::cout << " ch "  << channel[c];
            std::cout << " max " << vmax[c] << " at " << maxPt[c];
            maxMaxError[1] = max(maxMaxError[1], vmax[c]);
        }
        std::cout << std::endl;

        Mat backInterDiff = abs(mBackInter - mSrc);
        meanStdDev(backInterDiff, vmean, vdev);
        std::cout << "backInterDiff: mean " << vmean << " stddev " << vdev << std::endl;
        split(backInterDiff, chDiff);
        for(int c = 0; c < 3; c++)
        {
            minMaxLoc(chDiff[c], &vmin[c], &vmax[c],
                      &minPt[c], &maxPt[c]);
            std::cout << " ch "  << channel[c];
            std::cout << " max " << vmax[c] << " at " << maxPt[c];
            maxMaxError[2] = max(maxMaxError[2], vmax[c]);
        }
        std::cout << std::endl;

        Mat backInterGoldDiff = abs(mBackInter - mBackGold);
        meanStdDev(backInterGoldDiff, vmean, vdev);
        std::cout << "backInterGoldDiff: mean " << vmean << " stddev " << vdev << std::endl;
        split(backInterGoldDiff, chDiff);
        for(int c = 0; c < 3; c++)
        {
            minMaxLoc(chDiff[c], &vmin[c], &vmax[c],
                      &minPt[c], &maxPt[c]);
            std::cout << " ch "  << channel[c];
            std::cout << " max " << vmax[c] << " at " << maxPt[c];
            maxMaxError[3] = max(maxMaxError[3], vmax[c]);
        }
        std::cout << std::endl;

        Mat tmp = INT_DATA ? mGold : (TO_BGR ? mGold*256 : mGold+128);
        imwrite(format((dir + "noInter%03d.png").c_str(),  (TO_BGR ? l : blue)), tmp);
        tmp = INT_DATA ? mInter : (TO_BGR ? mInter*256 : mInter+128);
        imwrite(format((dir + "useInter%03d.png").c_str(), (TO_BGR ? l : blue)), tmp);
        tmp = INT_DATA ? (TO_BGR ? chInter[2] : chInter[1]) : (TO_BGR ? chInter[2]*256 : chInter[1]+128);
        imwrite(format((dir + "red%03d.png").c_str(),      (TO_BGR ? l : blue)), tmp);
        tmp = INT_DATA ? (mGold-mInter) : (TO_BGR ? (mGold-mInter)*256+128 : (mGold-mInter)+128);
        imwrite(format((dir + "diff%03d.png").c_str(),     (TO_BGR ? l : blue)), tmp);
        tmp = INT_DATA ? abs(mGold-mInter) : (TO_BGR ? abs(mGold-mInter)*256 : abs(mGold-mInter));
        imwrite(format((dir + "absdiff%03d.png").c_str(),  (TO_BGR ? l : blue)), tmp);
        tmp = INT_DATA ? backGoldDiff : (TO_BGR ? backGoldDiff+128 : backGoldDiff*256);
        imwrite(format((dir + "backgolddiff%03d.png").c_str(),  (TO_BGR ? l : blue)), tmp);
        tmp = INT_DATA ? backInterDiff : (TO_BGR ? backInterDiff+128 : backInterDiff*256);
        imwrite(format((dir + "backinterdiff%03d.png").c_str(), (TO_BGR ? l : blue)), tmp);

        if(randomFill)
        {
            RNG rng;
            for(size_t p = 0; p < pSize; p++)
            {
                float* pRow   = mSrc.ptr<float>(p);
                uchar* pRow_b = mSrc.ptr<uchar>(p);
                for(size_t q = 0; q < pSize; q++)
                {
                    if(INT_DATA)
                    {
                        if(TO_BGR)
                        {
                            //Lab
                            pRow_b[3*q + 0] = rng(256)*255/100;
                            pRow_b[3*q + 1] = rng(256);
                            pRow_b[3*q + 2] = rng(256);
                        }
                        else
                        {
                            //BGR
                            pRow_b[3*q + 0] = rng(256);
                            pRow_b[3*q + 1] = rng(256);
                            pRow_b[3*q + 2] = rng(256);
                        }
                    }
                    else
                    {
                        if(TO_BGR)
                        {
                            //Lab
                            pRow[3*q + 0] = (float)rng*100.0f;
                            pRow[3*q + 1] = 256.0f*(float)rng-128.0f;
                            pRow[3*q + 2] = 256.0f*(float)rng-128.0f;
                        }
                        else
                        {
                            //BGR
                            pRow[3*q + 0] = (float)rng;
                            pRow[3*q + 1] = (float)rng;
                            pRow[3*q + 2] = (float)rng;
                        }
                    }
                }
            }
        }

        //perf test
        std::cout << "perf: ";
        TickMeter tm; double t;
        //Lab to BGR
        tm.start();
        for(int i = 0; i < nPerfIters; i++)
        {
            for(size_t p = 0; p < pSize; p++)
            {
                float* pSrc   =   mSrc.ptr<float>(p);
                float* pInter = mInter.ptr<float>(p);
                uchar* pSrc_b   =   mSrc.ptr<uchar>(p);
                uchar* pInter_b = mInter.ptr<uchar>(p);
                if(INT_DATA)
                {
                    interToBgr_b(pSrc_b, pInter_b, pSize);
                }
                else
                {
                    interToBgr(pSrc, pInter, pSize);
                }
            }
        }
        tm.stop();
        t = tm.getTimeSec(); times[0] += t;
        std::cout << "inter lab2bgr: " << t << " ";
        tm.reset(); tm.start();
        for(int i = 0; i < nPerfIters; i++)
        {
            for(size_t p = 0; p < pSize; p++)
            {
                float* pSrc   =   mSrc.ptr<float>(p);
                float* pGold = mGold.ptr<float>(p);
                uchar* pSrc_b  =  mSrc.ptr<uchar>(p);
                uchar* pGold_b = mGold.ptr<uchar>(p);
                if(INT_DATA)
                {
                    goldToBgr_b(pSrc_b, pGold_b, pSize);
                }
                else
                {
                    goldToBgr(pSrc, pGold, pSize);
                }
            }
        }
        tm.stop();
        t = tm.getTimeSec(); times[1] += t;
        std::cout << "gold lab2bgr: " << t << " ";
        //RGB to Lab
        tm.reset(); tm.start();
        for(int i = 0; i < nPerfIters; i++)
        {
            for(size_t p = 0; p < pSize; p++)
            {
                float* pSrc   =   mSrc.ptr<float>(p);
                float* pInter = mInter.ptr<float>(p);
                uchar* pSrc_b   =   mSrc.ptr<uchar>(p);
                uchar* pInter_b = mInter.ptr<uchar>(p);
                if(INT_DATA)
                {
                    interToLab_b(pSrc_b, pInter_b, pSize);
                }
                else
                {
                    interToLab(pSrc, pInter, pSize);
                }
            }
        }
        tm.stop();
        t = tm.getTimeSec(); times[2] += t;
        std::cout << "inter rgb2lab: " << t << " ";
        tm.reset(); tm.start();
        for(int i = 0; i < nPerfIters; i++)
        {
            for(size_t p = 0; p < pSize; p++)
            {
                float* pSrc   =   mSrc.ptr<float>(p);
                float* pGold = mGold.ptr<float>(p);
                uchar* pSrc_b  =  mSrc.ptr<uchar>(p);
                uchar* pGold_b = mGold.ptr<uchar>(p);
                if(INT_DATA)
                {
                    goldToLab_b(pSrc_b, pGold_b, pSize);
                }
                else
                {
                    goldToLab(pSrc, pGold, pSize);
                }
            }
        }
        tm.stop();
        t = tm.getTimeSec(); times[3] += t;
        std::cout << "gold rgb2lab: " << t << " ";
        std::cout << std::endl;
        count++;
    }

    //max-max channel errors
    std::cout << std::endl << (TO_BGR ? "Lab2RGB" : "RGB2Lab") << " ";
    std::cout << "lab_lut_shift " << (int)lab_lut_shift << " ";
    for(int i = 0; i < 4; i++)
    {
        std::cout << maxMaxError[i] << "\t";
    }
    std::cout << std::endl;

    //overall perf
    for(int i = 0; i < 4; i++)
    {
        times[i] /= count;
    }
    std::cout << "perf: ";
    std::cout << "inter lab2bgr: " << times[0] << " ";
    std::cout << "gold lab2bgr: "  << times[1] << " ";
    std::cout << "inter rgb2lab: " << times[2] << " ";
    std::cout << "gold rgb2lab: "  << times[3] << " ";
    std::cout << std::endl;
}
