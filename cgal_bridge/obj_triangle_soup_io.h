#pragma once

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/number_utils.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bridge {

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = Kernel::Point_3;
using Triangle = std::array<std::size_t, 3>;

struct ObjSoup {
  std::vector<Point_3> points;
  std::vector<Triangle> triangles;
};

inline std::string trim_comment(std::string line) {
  const std::size_t hash_pos = line.find('#');
  if (hash_pos != std::string::npos) {
    line = line.substr(0, hash_pos);
  }
  const std::size_t first = line.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return std::string();
  }
  const std::size_t last = line.find_last_not_of(" \t\r\n");
  return line.substr(first, last - first + 1);
}

inline std::size_t parse_obj_vertex_index(const std::string& token, std::size_t vertex_count) {
  const std::size_t slash = token.find('/');
  const std::string head = token.substr(0, slash);
  if (head.empty()) {
    throw std::runtime_error("Malformed OBJ face token: '" + token + "'");
  }
  const long long raw = std::stoll(head);
  if (raw > 0) {
    const std::size_t idx = static_cast<std::size_t>(raw - 1);
    if (idx >= vertex_count) {
      throw std::runtime_error("OBJ face vertex index out of range: '" + token + "'");
    }
    return idx;
  }
  if (raw < 0) {
    const long long resolved = static_cast<long long>(vertex_count) + raw;
    if (resolved < 0 || resolved >= static_cast<long long>(vertex_count)) {
      throw std::runtime_error("OBJ negative face vertex index out of range: '" + token + "'");
    }
    return static_cast<std::size_t>(resolved);
  }
  throw std::runtime_error("OBJ vertex index 0 is invalid");
}

inline ObjSoup load_obj_triangle_soup(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Unable to open OBJ file: " + path.string());
  }

  ObjSoup soup;
  std::string line;
  std::size_t line_no = 0;
  while (std::getline(input, line)) {
    ++line_no;
    line = trim_comment(line);
    if (line.empty()) {
      continue;
    }

    std::istringstream iss(line);
    std::string head;
    iss >> head;
    if (head == "v") {
      double x = 0.0;
      double y = 0.0;
      double z = 0.0;
      if (!(iss >> x >> y >> z)) {
        throw std::runtime_error("Invalid vertex record at " + path.string() + ":" + std::to_string(line_no));
      }
      soup.points.emplace_back(x, y, z);
      continue;
    }

    if (head == "f") {
      std::vector<std::size_t> polygon;
      std::string token;
      while (iss >> token) {
        polygon.push_back(parse_obj_vertex_index(token, soup.points.size()));
      }
      if (polygon.size() < 3) {
        throw std::runtime_error("Face with fewer than 3 vertices at " + path.string() + ":" + std::to_string(line_no));
      }
      for (std::size_t i = 1; i + 1 < polygon.size(); ++i) {
        soup.triangles.push_back(Triangle{polygon[0], polygon[i], polygon[i + 1]});
      }
      continue;
    }

    // First bridge version intentionally ignores normals / UVs / materials / groups / smoothing.
  }

  return soup;
}

inline void save_obj_triangle_soup(const std::filesystem::path& path,
                                   const std::vector<Point_3>& points,
                                   const std::vector<Triangle>& triangles) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("Unable to write OBJ file: " + path.string());
  }
  for (const Point_3& p : points) {
    output << "v "
           << CGAL::to_double(p.x()) << ' '
           << CGAL::to_double(p.y()) << ' '
           << CGAL::to_double(p.z()) << '\n';
  }
  for (const Triangle& tri : triangles) {
    output << "f " << (tri[0] + 1) << ' ' << (tri[1] + 1) << ' ' << (tri[2] + 1) << '\n';
  }
}

}  // namespace bridge
