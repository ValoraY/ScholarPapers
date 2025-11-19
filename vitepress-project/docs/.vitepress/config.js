import { defineConfig } from 'vitepress'

export default defineConfig(async () => {
  const fs = await import('fs')
  const path = await import('path')

  const authorsDir = path.resolve(process.cwd(), 'docs/authors')

  let authorItems = []

  if (fs.existsSync(authorsDir)) {
    const files = fs.readdirSync(authorsDir)
    authorItems = files
      .filter(f => f.endsWith('.md') && f !== 'index.md')
      .map(f => ({
        text: f.replace('.md', ''),
        link: '/authors/' + f.replace('.md', '')
      }))
  }

  return {
    title: "Scholar Papers",
    description: "自动化学术论文收集网站",

    themeConfig: {
      nav: [
        { text: '论文合集', link: '/authors/' }
      ],

      sidebar: {
        '/authors/': [
          {
            text: '作者列表',
            collapsible: true,    // 可折叠
            collapsed: false,   // 默认展开
            items: authorItems  // ✔ authors 列表
          }
        ]
      },

      outline: false,

      head: [
        ['link', { rel: 'stylesheet', href: '/custom.css' }]
      ]
    }
  }
})
