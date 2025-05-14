import { router } from '@kit.ArkUI'
import { UserDataManager } from '../utils/UserDataManager'

@Entry
@Component
struct Login {
  @State username: string = ''
  @State password: string = ''
  @State message: string = ''

  aboutToAppear() {
    // 确保默认账号存在
    UserDataManager.saveUser('admin', '123456')
  }

  build() {
    Stack() {
      // 背景图片
      Image($r('app.media.login_bg'))
        .width('100%')
        .height('100%')
        .objectFit(ImageFit.Cover)

      // 登录表单
      Column() {
        Text('欢迎登录')
          .fontSize(32)
          .fontWeight(FontWeight.Bold)
          .margin({ top: 100, bottom: 40 })
          .fontColor('#333333')

        TextInput({ placeholder: '请输入用户名' })
          .width('90%')
          .height(50)
          .margin({ bottom: 20 })
          .backgroundColor('rgba(255, 255, 255, 0.9)')
          .borderRadius(25)
          .padding({ left: 20, right: 20 })
          .placeholderColor('#999999')
          .onChange((value: string) => {
            this.username = value
          })

        TextInput({ placeholder: '请输入密码' })
          .width('90%')
          .height(50)
          .margin({ bottom: 30 })
          .backgroundColor('rgba(255, 255, 255, 0.9)')
          .borderRadius(25)
          .padding({ left: 20, right: 20 })
          .placeholderColor('#999999')
          .type(InputType.Password)
          .onChange((value: string) => {
            this.password = value
          })

        Button('登录')
          .width('90%')
          .height(50)
          .backgroundColor('#007DFF')
          .borderRadius(25)
          .fontSize(18)
          .fontWeight(FontWeight.Medium)
          .onClick(async () => {
            if (!this.username || !this.password) {
              this.message = '请输入用户名和密码'
              return
            }
            
            const isValid = await UserDataManager.verifyUser(this.username, this.password)
            if (isValid) {
              router.pushUrl({ url: 'pages/Main' })
            } else {
              this.message = '用户名或密码错误'
            }
          })

        Button('注册')
          .width('90%')
          .height(50)
          .margin({ top: 20 })
          .backgroundColor('#4CAF50')
          .borderRadius(25)
          .fontSize(18)
          .fontWeight(FontWeight.Medium)
          .onClick(() => {
            router.pushUrl({ url: 'pages/Register' })
          })

        if (this.message) {
          Text(this.message)
            .fontSize(16)
            .fontColor(Color.Red)
            .margin({ top: 20 })
        }

        // 默认账号提示
        Text('默认账号：admin\n默认密码：123456')
          .fontSize(14)
          .fontColor('#666666')
          .margin({ top: 30 })
          .textAlign(TextAlign.Center)
      }
      .width('100%')
      .height('100%')
      .backgroundColor('rgba(255, 255, 255, 0.7)')
    }
    .width('100%')
    .height('100%')
  }
} 