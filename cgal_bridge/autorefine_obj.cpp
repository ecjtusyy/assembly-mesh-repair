#include <CGAL/Polygon_mesh_processing/autorefinement.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "obj_triangle_soup_io.h"

namespace PMP = CGAL::Polygon_mesh_processing;

namespace {

struct Options {
  std::filesystem::path input_path;
  std::filesystem::path output_path;
  int snap_grid_size = 23;
  int number_of_iterations = 5;
};

void print_usage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " <input.obj> <output.obj> [--snap_grid_size N] [--number_of_iterations N]\n";
}

Options parse_options(int argc, char** argv) {
  if (argc < 3) {
    print_usage(argv[0]);
    std::exit(2);
  }

  Options opts;
  opts.input_path = argv[1];
  opts.output_path = argv[2];
  for (int i = 3; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--snap_grid_size") {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value after --snap_grid_size");
      }
      opts.snap_grid_size = static_cast<int>(std::stoi(argv[++i]));
    } else if (arg == "--number_of_iterations") {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value after --number_of_iterations");
      }
      opts.number_of_iterations = static_cast<int>(std::stoi(argv[++i]));
    } else {
      throw std::runtime_error("Unexpected argument: " + arg);
    }
  }

  if (opts.snap_grid_size >= 52) {
    throw std::runtime_error("snap_grid_size must be < 52");
  }
  if (opts.number_of_iterations <= 0) {
    throw std::runtime_error("number_of_iterations must be >= 1");
  }
  return opts;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opts = parse_options(argc, argv);

    bridge::ObjSoup soup = bridge::load_obj_triangle_soup(opts.input_path);
    if (soup.triangles.empty()) {
      throw std::runtime_error("Input OBJ contains no triangles");
    }

    const bool input_self_intersect = PMP::does_triangle_soup_self_intersect<CGAL::Sequential_tag>(
        soup.points, soup.triangles);
    std::cout << "input_self_intersect=" << (input_self_intersect ? 1 : 0) << '\n';
    std::cout << "input_points=" << soup.points.size() << '\n';
    std::cout << "input_triangles=" << soup.triangles.size() << '\n';

    const bool ok = PMP::autorefine_triangle_soup(
        soup.points,
        soup.triangles,
        CGAL::parameters::apply_iterative_snap_rounding(true)
            .snap_grid_size(opts.snap_grid_size)
            .number_of_iterations(opts.number_of_iterations));

    const bool output_self_intersect = PMP::does_triangle_soup_self_intersect<CGAL::Sequential_tag>(
        soup.points, soup.triangles);

    std::cout << "autorefine_success=" << (ok ? 1 : 0) << '\n';
    std::cout << "output_self_intersect=" << (output_self_intersect ? 1 : 0) << '\n';
    std::cout << "output_points=" << soup.points.size() << '\n';
    std::cout << "output_triangles=" << soup.triangles.size() << '\n';

    if (!ok) {
      std::cerr << "ERROR: CGAL autorefine_triangle_soup() did not certify the soup within the requested snap-rounding iterations.\n";
      return 3;
    }
    if (output_self_intersect) {
      std::cerr << "ERROR: Post-autorefine soup is still self-intersecting according to CGAL.\n";
      return 4;
    }

    bridge::save_obj_triangle_soup(opts.output_path, soup.points, soup.triangles);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << '\n';
    return 1;
  }
}
