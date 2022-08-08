//
// Created by test on 2022-08-08.
//
#include <fmt/chrono.h>
#include <vector>
#include <fmt/ranges.h>
#include <fmt/os.h>
#include <fmt/color.h>

int main() {

  //example 1
  using namespace std::literals::chrono_literals;
  fmt::print("Default format: {} {}\n", 42s, 100ms);
  fmt::print("strftime-like format: {:%H:%M:%S}\n", 3h + 15min + 30s);

  //example 2
  std::vector<int> v = {1, 2, 3};
  fmt::print("{}\n", v);

  //example 3
  auto out = fmt::output_file("guide.txt");
  out.print("Don't {}", "Panic");

  //example 4
  fmt::print(fg(fmt::color::crimson) | fmt::emphasis::bold,
             "Hello, {}!\n", "world");
  fmt::print(fg(fmt::color::floral_white) | bg(fmt::color::slate_gray) |
      fmt::emphasis::underline, "Hello, {}!\n", "мир");
  fmt::print(fg(fmt::color::steel_blue) | fmt::emphasis::italic,
             "Hello, {}!\n", "世界");

  //example 5
  std::string s = fmt::format("I'd rather be {1} than {0}.", "right", "happy");
}