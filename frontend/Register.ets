import { router } from '@kit.ArkUI'
import { UserDataManager } from '../utils/UserDataManager'

@Entry
@Component
struct Register {
  @State username: string = ''
  @State password: string = ''
  @State confirmPassword: string = ''
  @State message: string = ''
  @State isPasswordVisible: boolean = false
  @State isConfirmPasswordVisible: boolean = false

  build() {
    Stack() {
      // 背景图片
      Image($r('app.media.login_bg'))
        .width('100%')
        .height('100%')
        .objectFit(ImageFit.Cover)

      // 返回按钮
      Button({ type: ButtonType.Circle }) {
        Image($r('app.media.startIcon'))
          .width(24)
          .height(24)
      }
      .width(40)
      .height(40)
      .position({ x: 20, y: 50 })
      .backgroundColor('rgba(255, 255, 255, 0.8)')
      .onClick(() => {
        router.back()
      })

      // 注册表单
      Column() {
        // 顶部装饰图片
        Image($r('app.media.foreground'))
          .width(120)
          .height(120)
          .margin({ top: 40, bottom: 20 })
          .animation({
            duration: 1000,
            curve: Curve.EaseInOut,
            iterations: -1,
            playMode: PlayMode.Alternate
          })

        // 标题区域
        Column() {
          Text('创建账号')
            .fontSize(36)
            .fontWeight(FontWeight.Bold)
            .fontColor('#333333')
            .margin({ bottom: 10 })

          Text('请填写以下信息完成注册')
            .fontSize(16)
            .fontColor('#666666')
            .margin({ bottom: 30 })
        }
        .width('100%')
        .alignItems(HorizontalAlign.Center)

        // 输入框区域
        Column() {
          // 用户名输入框
          Column() {
            Row() {
              Image($r('app.media.startIcon'))
                .width(20)
                .height(20)
                .margin({ right: 8 })
              Text('用户名')
                .fontSize(14)
                .fontColor('#666666')
            }
            .alignSelf(ItemAlign.Start)
            .margin({ left: 20, bottom: 8 })
            
            TextInput({ placeholder: '请输入用户名' })
              .width('90%')
              .height(50)
              .backgroundColor('rgba(255, 255, 255, 0.9)')
              .borderRadius(25)
              .padding({ left: 20, right: 20 })
              .placeholderColor('#999999')
              .onChange((value: string) => {
                this.username = value
              })
          }
          .margin({ bottom: 20 })

          // 密码输入框
          Column() {
            Row() {
              Image($r('app.media.foreground'))
                .width(20)
                .height(20)
                .margin({ right: 8 })
              Text('密码')
                .fontSize(14)
                .fontColor('#666666')
            }
            .alignSelf(ItemAlign.Start)
            .margin({ left: 20, bottom: 8 })
            
            Row() {
              TextInput({ placeholder: '请输入密码' })
                .width('85%')
                .height(50)
                .backgroundColor('rgba(255, 255, 255, 0.9)')
                .borderRadius(25)
                .padding({ left: 20, right: 20 })
                .placeholderColor('#999999')
                .type(this.isPasswordVisible ? InputType.Normal : InputType.Password)
                .onChange((value: string) => {
                  this.password = value
                })

              Button({ type: ButtonType.Circle }) {
                Image(this.isPasswordVisible ? $r('app.media.startIcon') : $r('app.media.foreground'))
                  .width(20)
                  .height(20)
              }
              .width(40)
              .height(40)
              .backgroundColor('rgba(255, 255, 255, 0.9)')
              .onClick(() => {
                this.isPasswordVisible = !this.isPasswordVisible
              })
            }
            .width('90%')
            .justifyContent(FlexAlign.SpaceBetween)
          }
          .margin({ bottom: 20 })

          // 确认密码输入框
          Column() {
            Row() {
              Image($r('app.media.foreground'))
                .width(20)
                .height(20)
                .margin({ right: 8 })
              Text('确认密码')
                .fontSize(14)
                .fontColor('#666666')
            }
            .alignSelf(ItemAlign.Start)
            .margin({ left: 20, bottom: 8 })
            
            Row() {
              TextInput({ placeholder: '请再次输入密码' })
                .width('85%')
                .height(50)
                .backgroundColor('rgba(255, 255, 255, 0.9)')
                .borderRadius(25)
                .padding({ left: 20, right: 20 })
                .placeholderColor('#999999')
                .type(this.isConfirmPasswordVisible ? InputType.Normal : InputType.Password)
                .onChange((value: string) => {
                  this.confirmPassword = value
                })

              Button({ type: ButtonType.Circle }) {
                Image(this.isConfirmPasswordVisible ? $r('app.media.startIcon') : $r('app.media.foreground'))
                  .width(20)
                  .height(20)
              }
              .width(40)
              .height(40)
              .backgroundColor('rgba(255, 255, 255, 0.9)')
              .onClick(() => {
                this.isConfirmPasswordVisible = !this.isConfirmPasswordVisible
              })
            }
            .width('90%')
            .justifyContent(FlexAlign.SpaceBetween)
          }
        }
        .width('100%')
        .margin({ top: 20 })

        // 注册按钮
        Button('注册')
          .width('90%')
          .height(50)
          .backgroundColor('#4CAF50')
          .borderRadius(25)
          .fontSize(18)
          .fontWeight(FontWeight.Medium)
          .margin({ top: 40 })
          .onClick(async () => {
            if (!this.username || !this.password || !this.confirmPassword) {
              this.message = '请填写完整信息'
              return
            }
            if (this.password !== this.confirmPassword) {
              this.message = '两次输入的密码不一致'
              return
            }
            
            // 保存用户数据
            const success = await UserDataManager.saveUser(this.username, this.password)
            if (success) {
              this.message = '注册成功'
              setTimeout(() => {
                router.pushUrl({ url: 'pages/Login' })
              }, 1500)
            } else {
              this.message = '注册失败，请重试'
            }
          })

        if (this.message) {
          Text(this.message)
            .fontSize(16)
            .fontColor(this.message === '注册成功' ? Color.Green : Color.Red)
            .margin({ top: 20 })
            .animation({
              duration: 300,
              curve: Curve.EaseInOut
            })
        }

        // 底部提示
        Row() {
          Image($r('app.media.startIcon'))
            .width(16)
            .height(16)
            .margin({ right: 8 })
          Text('已有账号？点击返回登录')
            .fontSize(14)
            .fontColor('#666666')
        }
        .margin({ top: 20, bottom: 40 })
        .onClick(() => {
          router.back()
        })
      }
      .width('100%')
      .height('100%')
      .backgroundColor('rgba(255, 255, 255, 0.7)')
    }
    .width('100%')
    .height('100%')
  }
} 