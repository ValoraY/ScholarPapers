# 📘 Scholar Papers — Google Scholar 自动爬取 & VitePress 静态站点

本项目用于自动化抓取指定学者的 **Google Scholar 论文**，并自动生成 Markdown 文件，同时利用 **VitePress** 自动构建成可公开访问的静态网站。

系统支持：

* ✅ **自动抓取最新论文**
* ✅ **增量更新（仅抓新增论文）**
* ✅ **按年份过滤论文**
* ✅ **为每位学者生成独立页面**
* ✅ **自动构建 VitePress 网站**
* ✅ **自动部署到 GitHub Pages**
* ✅ **支持 GitHub Actions 定时运行（每周一次）**
* ✅ **支持 GitHub Variables 远程配置学者列表与年份**


# 📂 项目目录结构

```
ScholarPapers/
├── config.json                      # 默认配置
├── config_override.json             # GitHub Actions 自动生成（优先级更高）
├── data/
│   └── author_jsons/                # 每位学者的 JSON 数据缓存（用于增量爬取）
│       └── zhangsan_xxxx.json
├── scripts/
│   └── fetch_papers.py              # 主爬虫脚本（抓取 + 生成 Markdown）
├── vitepress-project/               # VitePress 网站
│   ├── docs/
│   │   ├── index.md
│   │   └── authors/                 # 各学者页面（自动生成）
│   │       ├── zhangsan.md
│   ├── package.json
│   ├── .vitepress/config.js         # 动态生成作者列表
│   └── ...
└── .github/
    └── workflows/
        └── update.yml               # 自动爬取 + 自动构建 + GitHub Pages 部署
```


# 🚀 功能介绍

## 🔄 1. 自动抓取 Google Scholar 数据

基于 `scholarly` 库，可抓取：

* 论文标题
* 发表年份
* 摘要（Google Scholar 截断时自动从 arXiv 获取完整摘要）
* 论文链接

程序会自动执行：

* ✔ 按年份过滤
* ✔ 避免重复（增量更新）
* ✔ 自动保存到 JSON 文件
* ✔ 自动生成作者 Markdown 页面


## 📝 2. VitePress 自动生成学术站点

作者页面会自动放置在：

```
vitepress-project/docs/authors/*.md
```

主页与侧边栏菜单由：

```
.vitepress/config.js
```

自动生成，不需要手工维护作者列表。

最终构建的静态网站会发布到：

👉 **GitHub Pages**


# ⚙️ 配置方式

系统支持两种配置方式：


# 🟦 方式 1（推荐）：GitHub Variables 动态配置

前往：

```
GitHub → Settings → Secrets and Variables → Actions → Variables
```

添加下面这些变量：

| 变量名                 | 示例值                                              | 说明            |
| ------------------- | ------------------------------------------------ | ------------- |
| `SCHOLAR_AUTHORS`   | `[{"name": "zhangsan", "id": "BxxxxxAAAAJ"}]` | 学者数组（JSON 格式） |
| `YEAR_START`        | `2020`                                           | 最早抓取年份        |
| `INCREMENTAL_LIMIT` | `20`                                             | 增量模式每次抓取论文数量  |

### 🔧 示例（SCHOLAR_AUTHORS）

```json
[
  { "name": "zhangsan", "id": "BxxxxxAAAAJ" },
  { "name": "AnotherAuthor", "id": "XXXXXAAAAJ" }
]
```

GitHub Actions 会自动创建：
`config_override.json` → 覆盖 `config.json`


# 🟩 方式 2：本地 config.json（默认配置）

```json
{
  "year_start": 2020,
  "incremental_limit": 20,
  "authors": [
    { "name": "zhangsan", "id": "BxxxxxAAAAJ" }
  ]
}
```

若存在 `config_override.json`，则优先使用覆盖配置。

---

# 🤖 GitHub Actions 自动任务

自动任务配置位于：

```
.github/workflows/update.yml
```

系统功能包括：

* ✔ 每周自动抓取（CRON）
* ✔ 自动生成 Markdown
* ✔ 自动生成 VitePress 静态网站
* ✔ 自动提交更新
* ✔ 自动部署 GitHub Pages

你也可以手动触发：

👉 GitHub → Actions → Update Scholar Papers → **Run workflow**


# 🖥️ 本地运行方式

安装依赖：

```
pip install scholarly requests beautifulsoup4 httpx[socks]
```

执行爬虫：

```
python scripts/fetch_papers.py
```

生成的作者页面在：

```
vitepress-project/docs/authors/
```

本地预览 VitePress：

```
cd vitepress-project
npm install
npm run dev
```

打开：

👉 [http://localhost:5173](http://localhost:5173)


# 🆕 增量更新逻辑

程序自动检测：

### ✔ 首次运行

抓取 **最多 200 篇论文**

### ✔ 后续运行

只抓取最新 **N 篇论文**（默认 N=20，可配置）

并根据 JSON 缓存自动去重。

# 📜 摘要处理策略

* Google Scholar 若出现 `...` 截断
* 程序会自动尝试从 arXiv API 获取完整摘要
* 若非 arXiv 或无完整摘要 → 使用原始摘要



# 📌 注意事项

* Google Scholar 有访问限制，爬取速度较慢
* 若过于频繁请求可能被 captcha 阻断（脚本会自动重试）
* `data/author_jsons/*.json` 不要手动修改
* 修改学者列表推荐使用 GitHub Variables


# 🎉 总结

本项目为学术论文管理提供了：

* 自动化
* 无需服务器（完全依赖 GitHub Actions）
* 可自定义作者配置
* 增量爬取 + arXiv 补全
* 自带完整静态网站（VitePress）
* 自动部署 GitHub Pages

**轻量、稳定、免维护，是构建学术论文更新站点的极佳选择！**
