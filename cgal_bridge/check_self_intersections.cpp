#include <CGAL/Polygon_mesh_processing/self_intersections.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "obj_triangle_soup_io.h"

namespace PMP = CGAL::Polygon_mesh_processing;

namespace {

void print_usage(const char* argv0) {
  std::cerr << "Usage: " << argv0 << " <input.obj> [--list_pairs]\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      print_usage(argv[0]);
      return 2;
    }

    std::string input_path;
    bool list_pairs = false;
    for (int i = 1; i < argc; ++i) {
      const std::string arg(argv[i]);
      if (arg == "--list_pairs") {
        list_pairs = true;
      } else if (input_path.empty()) {
        input_path = arg;
      } else {
        std::cerr << "Unexpected argument: " << arg << "\n";
        print_usage(argv[0]);
        return 2;
      }
    }
    if (input_path.empty()) {
      print_usage(argv[0]);
      return 2;
    }

    const bridge::ObjSoup soup = bridge::load_obj_triangle_soup(input_path);

    const bool has_self_intersection = PMP::does_triangle_soup_self_intersect<CGAL::Sequential_tag>(
        soup.points, soup.triangles);

    std::vector<std::pair<std::size_t, std::size_t>> pairs;
    if (has_self_intersection) {
      PMP::triangle_soup_self_intersections<CGAL::Sequential_tag>(
          soup.points, soup.triangles, std::back_inserter(pairs));
    }

    std::cout << "self_intersect=" << (has_self_intersection ? 1 : 0) << '\n';
    std::cout << "count=" << pairs.size() << '\n';
    if (list_pairs) {
      for (const auto& pair : pairs) {
        std::cout << "pair=" << pair.first << ',' << pair.second << '\n';
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << '\n';
    return 1;
  }
}
