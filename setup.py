from setuptools import setup, find_packages

setup(name="ctrm",
      version="0.0.1",
      description="ctrms: cooperative timed roadmaps",
      author="Keisuke Okumura, Ryo Yonetani",
      author_email="okumura.k@coord.c.titech.ac.jp, ryo.yonetani@sinicx.com",
      python_requires=">=3.7",
      packages=["ctrm"],
      package_dir = {"": "src"},
      package_data={
          "ctrm": ["py.typed"],
      },
      install_requires=[],  # check Dockerfile
      )
