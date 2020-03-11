#include "injectors/smoke_injector2.h"
#include "smoke_solver2.h"
#include <render.h>

using namespace hermes::cuda;

#define WIDTH 800
#define HEIGHT 800

int main(int argc, char **argv) {
  int resSize = 128;
  if (argc > 1)
    sscanf(argv[1], "%d", &resSize);
  // sim
  ponos::size2 res(resSize, resSize);
  SmokeSolver2 solver;
  solver.setSpacing(ponos::vec2f(1.f / res.width, 1.f / res.height));
  solver.setResolution(res);
  solver.init();
  solver.rasterColliders();
  SphereSmokeInjector2 injector(point2(0, 0), 0.5);
  injector.inject(solver.scalarField(0), 0.1);
  // app
  circe::SceneApp<> app(WIDTH, HEIGHT, "Example", false);
  app.addViewport2D(0, 0, WIDTH, HEIGHT);
  app.getCamera<circe::UserCamera2D>(0)->fit(ponos::bbox2::unitBox());
  // cuda interop
  CudaOpenGLInterop cgl(res);
  // vis
  circe::ScreenQuad screen;
  screen.shader->begin();
  screen.shader->setUniform("tex", 0);
  app.renderCallback = [&]() {
    solver.step(0.01);
    renderDensity(solver.scalarField(0), cgl.bufferPointer<unsigned int>());
    renderSolids(solver.solid(), cgl.bufferPointer<unsigned int>());
    cgl.sendToTexture();
    cgl.bindTexture(GL_TEXTURE0);
    screen.render();
  };
  app.keyCallback = [&](int key, int scancode, int action, int modifiers) {
    if (action == GLFW_RELEASE) {
      if (key == GLFW_KEY_Q)
        app.exit();
      if (key == GLFW_KEY_SPACE) {
        // solver.step(0.01);
        // std::cerr << solver.scalarField(0).data() << std::endl;
      }
    }
  };
  circe::CartesianGrid grid(5);
  app.scene.add(&grid);
  app.run();
  return 0;
}