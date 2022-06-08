/*
// Copyright (c) 2019-2020 Ben Ashbaugh
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
*/

#pragma once
#include <fstream>
#include <stdint.h>

namespace BMP
{

#pragma pack(push, 1)

struct BMPFileHeader {
    uint16_t bf_type_; // 'BM' for BitMap
    uint32_t bf_size_; // file size in bytes
    uint16_t bf_reserved1_;
    uint16_t bf_reserved2_;
    uint32_t bf_off_bits_; // offset of bitmap in file
};

// Notes:
//  * The height of the bitmap is positive for bottom-to-top pixel order
//    or negative for top-to-bottom pixel order.
//  * Each row of pixel data must be a multiple of four bytes.
struct BMPInfoHeader {
    uint32_t bi_size_;      // info header size in bytes
    int32_t bi_width_;      // width of bitmap in pixels
    int32_t bi_height_;     // height of bitmap in pixels
    uint16_t bi_planes_;    // number of color planes
    uint16_t bi_bit_count_; // bit depth
    uint32_t bi_compression_;   // set to zero (BI_RGB) for no compression
    uint32_t bi_size_image_;// pixel data size in bytes, with padding
    int32_t bi_x_pels_per_meter_;
    int32_t bi_y_pels_per_meter_;
    uint32_t bi_clr_used_;
    uint32_t bi_clr_important_;
};

#pragma pack(pop)

// Writes four channel uint32_t data as a 32bpp RGBA BMP file
static bool save_image(
    const uint32_t *ptr, size_t width, size_t height,
    const char *file_name)
{
    std::ofstream os(file_name, std::ios::binary);
    if (!os.good())
        return false;

    // There is no alignment when writing 32bpp data.
    const size_t rowLength = width * 4;

    BMPFileHeader file_header = {0};
    BMPInfoHeader info_header = {0};

    file_header.bf_type_ = 0x4D42; // 'BM'
    file_header.bf_size_ = static_cast<uint32_t>(
        sizeof(file_header) + sizeof(info_header) + rowLength * height);
    file_header.bf_off_bits_ = sizeof(file_header) + sizeof(info_header);
    os.write(reinterpret_cast<const char*>(&file_header), sizeof(file_header));

    info_header.bi_size_ = sizeof(BMPInfoHeader);
    info_header.bi_width_ = static_cast<uint32_t>(width);
    info_header.bi_height_ = static_cast<uint32_t>(height);
    info_header.bi_planes_ = 1;
    info_header.bi_bit_count_ = 32;
    info_header.bi_compression_ = 0; // BI_RGB
    info_header.bi_size_image_ = static_cast<uint32_t>(rowLength * height);
    os.write(reinterpret_cast<const char*>(&info_header), sizeof(info_header));

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const uint32_t* ppix = ptr + (height - 1 - y) * width + x;
            os.write(reinterpret_cast<const char*>(ppix), 4);
        }
        // No alignment when writing 32bpp data.
    }

    return true;
}

// Writes single channel uint8_t data as a 24bpp RGB file
static bool save_image(
    const uint8_t *ptr, size_t width, size_t height,
    const char *file_name)
{
    std::ofstream os(file_name, std::ios::binary);
    if (!os.good())
        return false;

    const size_t rowLength = (width * 3 + (4 - 1)) & ~(4 - 1);
    const size_t align_size = rowLength - width * 3;

    BMPFileHeader file_header = {0};
    BMPInfoHeader info_header = {0};

    file_header.bf_type_ = 0x4D42; // 'BM'
    file_header.bf_size_ = static_cast<uint32_t>(
        sizeof(file_header) + sizeof(info_header) + rowLength * height);
    file_header.bf_off_bits_ = sizeof(file_header) + sizeof(info_header);
    os.write(reinterpret_cast<const char*>(&file_header), sizeof(file_header));

    info_header.bi_size_ = sizeof(BMPInfoHeader);
    info_header.bi_width_ = static_cast<uint32_t>(width);
    info_header.bi_height_ = static_cast<uint32_t>(height);
    info_header.bi_planes_ = 1;
    info_header.bi_bit_count_ = 8 * 3;
    info_header.bi_compression_ = 0; // BI_RGB
    info_header.bi_size_image_ = static_cast<uint32_t>(rowLength * height);
    os.write(reinterpret_cast<const char*>(&info_header), sizeof(info_header));

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const uint8_t* ppix = ptr + (height - 1 - y) * width + x;
            os.write(reinterpret_cast<const char*>(ppix), 1);
            os.write(reinterpret_cast<const char*>(ppix), 1);
            os.write(reinterpret_cast<const char*>(ppix), 1);
        }
        for (uint32_t a = 0; a < align_size; a++) {
            uint8_t padding = 0;
            os.write(reinterpret_cast<const char*>(&padding), 1);
        }
    }

    return true;
}

}
