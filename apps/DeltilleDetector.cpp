/**
* Copyright (C) 2017-present, Facebook, Inc.
*
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version.
*
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public
* License along with this library; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#include <deltille/target_detector.h>

#include <opencv2/highgui.hpp>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>


namespace fs = boost::filesystem;
namespace po = boost::program_options;

using namespace std;

/**
 */
bool writeCornersToFile(std::ostream &os,
                        const std::vector<CalibrationCorner> &corners,
                        string filename, const cv::Size &image_size,
                        bool write_ordered_only = true) {

  auto num_corners = corners.size();
  if(write_ordered_only) {
    num_corners = 0;
    for(auto& c : corners) {
      num_corners += c.isValid() && c.isOrdered;
    }
  }

  cout << "Writing " << num_corners << " corners to : " << filename << endl;

  os << "filename: " << filename << endl;
  os << "width: " << image_size.width << endl;
  os << "height: " << image_size.height << endl;
  os << "num_corners: " << num_corners << endl;
  os << "encoding: ascii" << endl;

  auto p = os.precision();
  os.precision(numeric_limits<double>::max_digits10);
  for (auto &c : corners) {
    if (!c.isValid() || (write_ordered_only && !c.isOrdered)) {
      continue;
    }

    os << c << endl;
  }

  os.precision(p);
  return os.good();
}

/**
 */
class DataSource {
public:
  virtual ~DataSource() {}

  bool getImage(cv::Mat &image, int index = -1) {
    if (this->get_image(image, index)) {
      convert_to_grayscale(image);
      convert_type(image);
      return true;
    } else {
      return false;
    }
  }

  const std::string &getLastFilename() const { return _last_file_name; }

private:
  virtual bool get_image(cv::Mat &, int) = 0;

  void convert_to_grayscale(cv::Mat &image) const {
    if (image.channels() == 3) {
      cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
      cv::cvtColor(image, image, cv::COLOR_BGRA2GRAY);
    }
  }

  void convert_type(cv::Mat &image) const {
    if (image.depth() == cv::DataDepth<uint16_t>::value) {
      double max_val = 0.0;
      cv::minMaxLoc(image, nullptr, &max_val);
      image.convertTo(image, CV_MAKETYPE(cv::DataDepth<float>::value, 1),
                      255.0 * (1.0 / max_val));
    }
  }

protected:
  string _last_file_name;
};

/**
 */
class ImageListDataSource : public DataSource {
public:
  ImageListDataSource(vector<string> &&filenames)
      : _filenames(move(filenames)) {}

private:
  bool get_image(cv::Mat &I, int f_i) override {
    if (f_i < 0)
      f_i = _counter++;

    if (std::size_t(f_i) < _filenames.size()) {
      this->_last_file_name = _filenames[f_i];
      return !(I = cv::imread(_filenames[f_i],
                              cv::IMREAD_ANYDEPTH | cv::IMREAD_GRAYSCALE))
                  .empty();
    } else {
      return false;
    }
  }

private:
  int _counter{0};
  vector<string> _filenames;
};

void RunDetector(DataSource *data_source, string target_dsc_fn,
                 const string output_dir, bool save_images) {
  TargetDetector target_detector(target_dsc_fn);
  
  cv::Mat I, output_image;
  for (int i = 0; data_source->getImage(I, i); ++i) {
    if (!I.empty()) {
      vector<CalibrationCorner> corners;
      target_detector.run(I, corners, save_images ? &output_image : nullptr);

      auto filename = data_source->getLastFilename();
      auto filepath = fs::path(filename);
      auto outpath =
          output_dir.empty() ? filepath.parent_path() : fs::path(output_dir);
      auto basename = filepath.stem();
      auto out_orpc_fn = outpath / fs::change_extension(basename, ".orpc");

      std::ofstream fo(out_orpc_fn.string());
      if (fo.is_open()) {
        writeCornersToFile(fo, corners, filename, I.size(), true);
      } else {
        cerr << "Failed to open: " << out_orpc_fn << " for writing"
              << endl;
      }

      if (save_images) {
        auto out_basename = "out_" + basename.string();
        auto out_img_fn = fs::change_extension(outpath / out_basename, ".png");
        cout << "Writing detection image to : " << out_img_fn << endl;
        cv::imwrite(out_img_fn.string(), output_image);
      }
    }
  }
}

int main(int argc, char **argv) {
  string target_dsc_fn;
  string output_dir{""};

  vector<string> files;

  po::options_description desc(argv[0]);
  desc.add_options()("help,h", "Produce this help message")(
      "target,t", po::value<string>(&target_dsc_fn)->required(),
      "Target *.dsc file")("files,f",
                           po::value<vector<string>>(&files)->multitoken(),
                           "List of image files")(
      "output,o", po::value<string>(&output_dir), "Output directory")(
      "save-images,s", "Store debug images");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  } catch (const exception &ex) {
    cerr << ex.what() << endl << endl;
    cerr << desc << endl;
    return 1;
  }

  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }

  if (!fs::exists(target_dsc_fn) || !fs::is_regular_file(target_dsc_fn)) {
    throw invalid_argument("invalid target *.dsc file '" + target_dsc_fn + "'");
  }

  if (vm.count("output")) {
    if (!fs::exists(output_dir)) {
      if (!fs::create_directory(output_dir)) {
        throw invalid_argument("invalid output directory " + output_dir);
      }
    } else {
      if (!fs::is_directory(output_dir)) {
        throw invalid_argument("argument to --output is NOT a directory");
      }
    }
  }

  if (!files.empty()) {
    ImageListDataSource data_source(move(files));
    RunDetector(&data_source, target_dsc_fn, output_dir, vm.count("save-images"));
  }
  cout << "end" << endl;
  return 0;
}
