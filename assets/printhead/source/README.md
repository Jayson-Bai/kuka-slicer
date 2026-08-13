# 喷头 CAD 来源文件

此目录是项目内保存的喷头资产来源副本：

- `干涉检查用喷嘴模块装配体.SLDASM`：SHA-256
  `64635f0d92bb400f4cce2d4aee54e0c310375b3452f0af8defc4f05c27414698`；
- `干涉检查用喷嘴模块装配体.STEP`：SHA-256
  `d76518e6e2aa005d197b5a905149cd3361ed5926b1681fe3cf516999998539e6`；
- `干涉检查用喷嘴模块装配体.STL`：SHA-256
  `6f6f662a3da5e705b6720e5204d6b0b8e244e21aa06f21bd13705602798da50f`。

运行时不会读取该文件或引用用户的 `F:` 目录；三维预览与碰撞预检只读取同级上层
目录中已生成的 GLB / preview JSON 网格资产。运行时资产由本目录的 STL 生成，STEP
用于复核 CAD 来源；要更新它们，运行 `scripts/build_printhead_asset.py`。
